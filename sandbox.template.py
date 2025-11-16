from math import *  # noqa: F403
from json import loads, dumps


def run(data: dict) -> dict:
    local_vars = {}
    exec(data["fn"], globals(), local_vars)

    try:
        func = None
        for name, obj in local_vars.items():
            if callable(obj):
                func = obj
                break

        if func is None:
            return {"error": "no function found in code"}

        result = func(**data["kwargs"])
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import sys

    data = loads(sys.stdin.read())
    print(dumps(run(data)))
