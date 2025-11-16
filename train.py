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
    .env({"PATH": "/root/unboxer/.venv/bin:$PATH"})
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
    gpu="H100:2",
    image=image,
    secrets=[openrouter, fly_api, postgres],
    timeout=14400,
)
def train_unboxer():
    import os
    import subprocess
    import time
    from trainer import train

    os.environ["NCCL_DEBUG"] = "WARN"
    os.environ["NCCL_SOCKET_IFNAME"] = "lo"

    print("starting vLLM inference server on GPU 0...")
    vllm_env = os.environ.copy()
    vllm_env["CUDA_VISIBLE_DEVICES"] = "0"
    vllm_process = subprocess.Popen(
        ["vf-vllm", "Qwen/Qwen3-0.6B"],
        env=vllm_env,
    )

    time.sleep(10)
    print("vLLM server should be starting up, beginning training on GPU 1...")
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    postgres_val = os.environ.get("POSTGRES", "NOT_SET")
    print(f"POSTGRES env var: {postgres_val[:50]}..." if len(postgres_val) > 50 else f"POSTGRES env var: {postgres_val}")

    try:
        result = train("configs/unboxer.toml")
        return result
    finally:
        print("terminating vLLM server...")
        vllm_process.terminate()
        vllm_process.wait(timeout=30)


@app.local_entrypoint()
def main():
    result = train_unboxer.remote()
    print(f"training complete: {result}")
