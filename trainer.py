#!/usr/bin/env python3
from pathlib import Path
import tomllib
import verifiers as vf
from os import environ


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

    env = vf.load_environment(env_id=env_id, **env_args)
    rl_config = vf.RLConfig(**config["trainer"].get("args", {}))
    trainer = vf.RLTrainer(model=model, env=env, args=rl_config)
    trainer.train()

    return {"status": "complete"}


if __name__ == "__main__":
    result = train()
    print(f"training complete: {result}")
