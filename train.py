import modal

app = modal.App("unboxer")

openrouter = modal.Secret.from_name("openrouter")
fly_api = modal.Secret.from_name("fly-api")
postgres = modal.Secret.from_name("postgres")
github = modal.Secret.from_name("github")

volume = modal.Volume.from_name("unboxer-volume")
VOLUME_DIR = "/mnt/data"
REPO_DIR = f"{VOLUME_DIR}/unboxer"

UV_SYNC_CMD = "uv sync --frozen --torch-backend cu128"

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "wget")
    .run_commands("pip install uv")
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "UV_HTTP_TIMEOUT": "600",
        "UV_CACHE_DIR": "/mnt/data/.uv-cache",
        "PATH": "/usr/local/cuda-12.8/bin:$PATH",
        "LD_LIBRARY_PATH": "/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH",
    })
    .run_commands(
        "wget -q https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl",
        "pip install --no-dependencies flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl",
    )
)


@app.function(
    gpu="H100",
    image=image,
    volumes={VOLUME_DIR: volume},
    secrets=[openrouter, fly_api, postgres, github],
    timeout=14400,
)
def train_unboxer():
    import sys
    import os
    import subprocess
    from pathlib import Path

    gh_token = os.environ.get("GH_TOKEN")
    if not gh_token:
        raise ValueError("GH_TOKEN not found in environment")

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

    print(f"installing dependencies: {UV_SYNC_CMD}")
    subprocess.run(UV_SYNC_CMD.split(), check=True)

    from trainer import train
    result = train("configs/unboxer.toml")
    volume.commit()
    return result


@app.local_entrypoint()
def main():
    result = train_unboxer.remote()
    print(f"training complete: {result}")
