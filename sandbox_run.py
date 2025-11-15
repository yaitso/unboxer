from math import *  # noqa: F403
from json import loads, dumps
from pathlib import Path


def run(data: dict) -> dict:
    exec(data["fn"], globals())
    try:
        if "blackbox" not in globals():
            return {"error": "function `blackbox` not defined"}
        result = blackbox(**data["kwargs"])  # noqa: F405  # pyright: ignore[reportUndefinedVariable]
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    data = loads(Path("/json").read_text())
    print(dumps(run(data)))
