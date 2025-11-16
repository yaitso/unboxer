#!/usr/bin/env python3
from pathlib import Path
import tomllib
import verifiers as vf
from os import environ
from huggingface_hub import HfApi, create_repo
import asyncio


async def run_migration():
    """run database migration once before training"""
    from db import RolloutsDB

    dsn = environ.get("POSTGRES")
    if dsn:
        await RolloutsDB.migrate(dsn=dsn)


def train(config_path: str = "configs/unboxer.toml") -> dict:
    """train model with given config"""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"config not found: {config_path}")

    with config_file.open("rb") as f:
        config = tomllib.load(f)

    model = config["model"]
    env_id = config["env"]["id"]
    env_args = config["env"].get("args", {})

    if "dsn" not in env_args and "POSTGRES" in environ:
        env_args["dsn"] = environ["POSTGRES"]

    if "train_commit" not in env_args and "TRAIN_COMMIT" in environ:
        env_args["train_commit"] = environ["TRAIN_COMMIT"]

    asyncio.run(run_migration())

    env = vf.load_environment(env_id=env_id, **env_args)
    rl_config = vf.RLConfig(**config["trainer"].get("args", {}))

    if "TRAIN_COMMIT" in environ:
        rl_config.run_name = environ["TRAIN_COMMIT"]

    trainer = vf.RLTrainer(model=model, env=env, args=rl_config)
    trainer.train()

    if "HF_TOKEN" in environ:
        hf_token = environ["HF_TOKEN"]
        hf_username = environ.get("HF_USERNAME", "yaitso")
        repo_name = "unboxer"
        repo_id = f"{hf_username}/{repo_name}"

        try:
            create_repo(repo_id=repo_id, token=hf_token, exist_ok=True)
        except Exception as e:
            print(f"repo creation warning: {e}")

        try:
            if hasattr(trainer.model, "push_to_hub_merged"):
                trainer.model.push_to_hub_merged(
                    repo_id,
                    trainer.tokenizer,
                    token=hf_token,
                    save_method="merged_16bit",
                )
                print(f"pushed model to {repo_id}")
            else:
                trainer.model.save_pretrained(rl_config.output_dir)
                trainer.tokenizer.save_pretrained(rl_config.output_dir)
                print(f"saved model to {rl_config.output_dir}")
        except Exception as e:
            print(f"model upload warning: {e}")

    return {"status": "complete"}


if __name__ == "__main__":
    result = train()
    print(f"training complete: {result}")
