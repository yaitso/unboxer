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
    .workdir("/root/unboxer")
    .run_commands(
        "uv venv",
        "uv pip install pip",
        "uv pip install --index-url https://download.pytorch.org/whl/cu128 torch",
        "wget -q https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl",
        "uv pip install --no-deps ./flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl",
        "uv sync --frozen --no-install-package flash-attn --no-install-package torch",
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
