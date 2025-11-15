from json import JSONDecodeError, loads, dumps
from subprocess import run
from result import Result, Ok, Err
from os import environ
from pathlib import Path


def build_sandbox_docker():
    script_dir = Path(__file__).parent
    run(
        ["docker", "build", "-t", "sandbox", "-f", "sandbox.Dockerfile", "."],
        cwd=script_dir,
    )


def build_wasm_docker():
    script_dir = Path(__file__).parent
    run(
        ["docker", "build", "-t", "wasm", "-f", "wasm.Dockerfile", "."],
        cwd=script_dir,
    )


def sandbox_docker(fn: str, kwargs: dict) -> Result[dict, str]:
    build_sandbox_docker()
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
    build_wasm_docker()
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


def sandbox_python(fn: str, kwargs: dict) -> Result[dict, str]:
    script_dir = Path(__file__).parent
    sandbox_run_code = (script_dir / "sandbox_run.py").read_text()

    payload = dumps({"fn": fn, "kwargs": kwargs})

    try:
        result = run(
            ["python3", "-c", sandbox_run_code],
            input=payload,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except Exception as e:
        return Err(f"python execution failed: {str(e)}")

    try:
        data = loads(result.stdout)
        if "error" in data:
            return Err(data["error"])
        return Ok({"output": data["result"]})
    except JSONDecodeError:
        return Err(
            f"json decode failed - stderr: {result.stderr}, stdout: {result.stdout}"
        )


def sandbox(fn: str, kwargs: dict) -> Result[dict, str]:
    mode = environ.get("SANDBOX_USE", "python")

    match mode:
        case "python":
            return sandbox_python(fn, kwargs)
        case "wasm":
            return sandbox_wasm(fn, kwargs)
        case "docker":
            return sandbox_docker(fn, kwargs)
        case _:
            return Err(f"unknown sandbox mode: {mode}")


if __name__ == "__main__":
    fn = """
def blackbox(a: float, b: float):
    return a + (sin(a)*b)**2
"""

    result = sandbox(fn, {"a": 1.0, "b": 2.0})

    if result.is_ok():
        print(dumps({"result": result.ok()}))
    else:
        print(dumps({"error": result.err()}))
