import modal

app = modal.App("unboxer")

openrouter = modal.Secret.from_name("openrouter")
fly_api = modal.Secret.from_name("fly-api")
postgres = modal.Secret.from_name("postgres")

volume = modal.Volume.from_name("unboxer-volume", create_if_missing=True)
VOLUME_DIR = "/root/volume"
REPO_DIR = f"{VOLUME_DIR}/unboxer"

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git")
    .run_commands("pip install uv")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "UV_HTTP_TIMEOUT": "600"})
)


@app.function(
    gpu="H100",
    image=image,
    volumes={VOLUME_DIR: volume},
    secrets=[openrouter, fly_api, postgres],
    timeout=14400,
)
def train_unboxer():
    import sys
    import os
    import subprocess
    from pathlib import Path

    os.environ["PATH"] = f"/usr/local/cuda-12.8/bin:{os.environ.get('PATH', '')}"
    os.environ["LD_LIBRARY_PATH"] = f"/usr/local/cuda-12.8/lib64:{os.environ.get('LD_LIBRARY_PATH', '')}"

    env_file = Path(f"{VOLUME_DIR}/.env")
    if not env_file.exists():
        raise FileNotFoundError(f"missing .env file at {env_file}")

    gh_token = None
    for line in env_file.read_text().splitlines():
        if line.startswith("GH_TOKEN="):
            gh_token = line.split("=", 1)[1].strip()
            break

    if not gh_token:
        raise ValueError("GH_TOKEN not found in .env")

    repo_path = Path(REPO_DIR)
    if not repo_path.exists():
        print(f"cloning repo to {REPO_DIR}...")
        subprocess.run([
            "git", "clone",
            f"https://{gh_token}@github.com/yaitso/rewarding.git",
            REPO_DIR
        ], check=True)
        volume.commit()
    else:
        print(f"pulling latest changes in {REPO_DIR}...")
        subprocess.run(["git", "-C", REPO_DIR, "pull"], env={**os.environ, "GH_TOKEN": gh_token}, check=True)
        volume.commit()

    os.chdir(REPO_DIR)
    sys.path.insert(0, REPO_DIR)

    print("installing dependencies on H100...")
    subprocess.run(["uv", "sync", "--frozen"], check=True)
    print("committing venv to volume...")
    volume.commit()

    from trainer import train
    result = train("configs/unboxer.toml")
    volume.commit()
    return result


@app.local_entrypoint()
def main():
    result = train_unboxer.remote()
    print(f"training complete: {result}")
