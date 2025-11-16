import modal

app = modal.App("unboxer")

openrouter = modal.Secret.from_name("openrouter")
fly_api = modal.Secret.from_name("fly-api")
postgres = modal.Secret.from_name("postgres")

volume = modal.Volume.from_name("unboxer-volume", create_if_missing=True)
VOLUME_DIR = "/workspace"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl", "ca-certificates", "build-essential")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "UV_HTTP_TIMEOUT": "600"})
    .uv_pip_install("huggingface_hub[hf_transfer]")
    .uv_pip_install("torch>=2.8.0", "triton>=3.4.0")
    .uv_pip_install(
        "transformers>=4.51.0",
        "vllm>=0.6.8",
        "click",
        "pytest",
        "pytest-asyncio",
        "result",
        "xxhash",
        "python-dotenv",
        "httpx>=0.28.1",
        "asyncpg>=0.30.0",
        "hypothesis>=6.148.0",
        "openai",
        "datasets",
        "verifiers[rl]",
    )
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
    from pathlib import Path
    import shutil

    try:
        import tomllib
    except ImportError:
        import tomli as tomllib

    import verifiers as vf

    os.makedirs("/root/unboxer", exist_ok=True)

    for item in [
        "environments",
        "sandbox.py",
        "db.py",
        "unboxer.py",
        "configs",
        "trainer.py",
    ]:
        src = Path(__file__).parent / item
        dst = Path("/root/unboxer") / item
        if src.is_file():
            shutil.copy2(src, dst)
        elif src.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)

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
