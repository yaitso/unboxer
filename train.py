import modal
from pathlib import Path

app = modal.App("unboxer")

openrouter = modal.Secret.from_name("openrouter")
fly_api = modal.Secret.from_name("fly-api")
postgres = modal.Secret.from_name("postgres")

volume = modal.Volume.from_name("unboxer-volume", create_if_missing=True)
VOLUME_DIR = "/workspace"

image = (
    modal.Image.from_registry("ghcr.io/astral-sh/uv:python3.12-bookworm-slim")
    .apt_install("git", "build-essential")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "UV_HTTP_TIMEOUT": "600"})
    .add_local_dir(
        ".",
        remote_path="/root/unboxer",
        copy=True,
        ignore=[
            "__pycache__",
            "*.pyc",
            ".git",
            ".venv",
            ".hypothesis",
            "*.log",
            "CONTEXT.md",
            "CONVO.md",
            "TODO.md",
            "old",
            "un.egg-info",
            "verifiers",
        ],
    )
    .workdir("/root/unboxer")
    .uv_sync()
)


@app.function(
    gpu="H100",
    image=image,
    volumes={VOLUME_DIR: volume},
    secrets=[openrouter, fly_api, postgres],
    timeout=14400,
)
def train_unboxer(config_path: str = "configs/unboxer.toml"):
    import sys
    import os

    try:
        import tomllib
    except ImportError:
        import tomli as tomllib

    import verifiers as vf

    os.chdir("/root/unboxer")
    sys.path.insert(0, "/root/unboxer")

    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"config not found: {config_path}")

    with config_file.open("rb") as f:
        config = tomllib.load(f)

    model = config["model"]
    env_id = config["env"]["id"]
    env_args = config["env"].get("args", {})

    env = vf.load_environment(env_id=env_id, **env_args)
    rl_config = vf.RLConfig(**config["trainer"].get("args", {}))
    trainer = vf.RLTrainer(model=model, env=env, args=rl_config)
    trainer.train()

    volume.commit()
    return {"status": "complete"}


@app.local_entrypoint()
def main(config: str = "configs/unboxer.toml"):
    result = train_unboxer.remote(config_path=config)
    print(f"training complete: {result}")
