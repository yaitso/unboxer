import modal

app = modal.App("test-fs")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "wget", "tree")
    .run_commands("pip install uv")
    .run_commands(
        "wget -q https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl",
        "pip install --no-dependencies flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl",
    )
    .run_commands("ls -la / > /tmp/rootfs.txt && cat /tmp/rootfs.txt")
)

@app.function(image=image)
def explore():
    import subprocess
    result = subprocess.run(["ls", "-la", "/"], capture_output=True, text=True)
    print("root filesystem:")
    print(result.stdout)
    return result.stdout

@app.local_entrypoint()
def main():
    result = explore.remote()
    print(f"\nfinal result:\n{result}")
