from json import JSONDecodeError, loads, dumps
from subprocess import run
from result import Result, Ok, Err
from os import environ
from pathlib import Path


def build():
    script_dir = Path(__file__).parent
    run(["docker", "build", "-t", "sandbox", "-f", "sandbox.Dockerfile", "."], cwd=script_dir)
    run(["docker", "build", "-t", "wasm", "-f", "wasm.Dockerfile", "."], cwd=script_dir)


def sandbox_docker(fn: str, kwargs: dict) -> Result[dict, str]:
    payload = dumps({"fn": fn, "kwargs": kwargs})

    try:
        result = run(
            [
                "docker",
                "run",
                "-i",
                "--rm",
                "--network=none",
                "--memory=100m",
                "--cpus=1",
                "--pids-limit=10",
                "--ulimit",
                "fsize=1048576:1048576",
                "--security-opt",
                "no-new-privileges",
                "--tmpfs",
                "/tmp:rw,noexec,nosuid,nodev,size=10m",
                "--cap-drop=ALL",
                "sandbox",
            ],
            input=payload,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except Exception as e:  # pragma: no cover
        return Err(f"docker execution failed: {str(e)}")
    try:
        data = loads(result.stdout)
        return Ok(data)
    except JSONDecodeError:  # pragma: no cover
        return Err(
            f"json decode failed - stderr: {result.stderr}, stdout: {result.stdout}"
        )


def sandbox_wasm(fn: str, kwargs: dict) -> Result[dict, str]:
    payload = dumps({"fn": fn, "kwargs": kwargs})

    try:
        result = run(
            [
                "docker",
                "run",
                "-i",
                "--rm",
                "--memory=100m",
                "--cpus=1",
                "wasm",
            ],
            input=payload,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception as e:
        return Err(f"wasm execution failed: {str(e)}")

    try:
        data = loads(result.stdout.strip())
        return Ok(data)
    except JSONDecodeError:
        return Err(
            f"json decode failed - stderr: {result.stderr}, stdout: {result.stdout}"
        )


def sandbox(fn: str, kwargs: dict) -> Result[dict, str]:
    mode = environ.get("SANDBOX_USE", "docker")

    if mode == "wasm":
        return sandbox_wasm(fn, kwargs)
    elif mode == "docker":
        return sandbox_docker(fn, kwargs)
    else:
        return Err(f"unknown sandbox mode: {mode}")


if __name__ == "__main__":
    build()

    fn = """
def blackbox(a: float, b: float):
    # snake activation function
    return a + (sin(a)*b)**2
"""

    result = sandbox(fn, {"a": 1.0, "b": 2.0})

    if result.is_ok():
        print(dumps({"result": result.ok()}))
    else:
        print(dumps({"error": result.err()}))
