import modal

app = modal.App("unboxer")

openrouter = modal.Secret.from_name("openrouter")
fly_api = modal.Secret.from_name("fly-api")
postgres = modal.Secret.from_name("postgres")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "wget")
    .run_commands("pip install uv")
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "UV_HTTP_TIMEOUT": "600",
            "PATH": "/usr/local/cuda-12.8/bin:$PATH",
            "LD_LIBRARY_PATH": "/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH",
        }
    )
    .add_local_file(
        "pyproject.toml",
        remote_path="/root/unboxer/pyproject.toml",
        copy=True,
    )
    .add_local_file(
        "uv.lock",
        remote_path="/root/unboxer/uv.lock",
        copy=True,
    )
    .workdir("/root/unboxer")
    .add_local_dir(
        "environments",
        remote_path="/root/unboxer/environments",
        copy=True,
    )
    .run_commands(
        "uv venv --python 3.12",
        "uv sync --frozen",
    )
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
            "proxy/target",
        ],
    )
)


@app.function(
    gpu="H100",
    image=image,
    secrets=[openrouter, fly_api, postgres],
    timeout=14400,
)
def train_unboxer():
    from trainer import train

    result = train("configs/unboxer.toml")
    return result


@app.local_entrypoint()
def main():
    result = train_unboxer.remote()
    print(f"training complete: {result}")
