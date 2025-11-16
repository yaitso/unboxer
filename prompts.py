import json


def complexity_json() -> str:
    return """{
  "num_ops": <int>,
  "num_holes": <int>,
  "num_args": <int>
}"""


def spec_json() -> str:
    return """{
  "fn": "def blackbox(a: float) -> float:\\n    return sin(a + $b$)",
  "kwargs": {"a": "st.floats(-10, 10)"},
  "holes": {"b": "st.floats(0, 5)"}
}

CONSTRAINTS:
- kwargs and holes MUST use st.ints(-10, 10) or st.floats(-10, 10)
- only bool, int, float builtins and `from math import *` allowed
- all values will be rounded to 1 decimal place"""


def complexity_explanation(complexity: dict, step: int) -> str:
    """explain complexity constraints based on train step"""
    examples = """
COUNTING RULES:
- each operator (+, -, *, /, **) counts as 1 op
- each math function (sin, cos, exp, sqrt, etc) counts as 1 op
- constants are holes if not 0 or 1
- arg reuse (x**x, y/y) is allowed and counts as 1 arg

EXAMPLES (num_ops, num_holes, num_args):
✓ def blackbox(x): return sin(x)         # 1 op, 0 holes, 1 arg
✓ def blackbox(y): return y**y           # 1 op, 0 holes, 1 arg
✓ def blackbox(x): return x + $a$        # 1 op, 1 hole, 1 arg
✓ def blackbox(x, y): return x + y       # 1 op, 0 holes, 2 args
✓ def blackbox(x, y): return x * sin(y)  # 2 ops, 0 holes, 2 args
✓ def blackbox(x): return (x + $a$) * $b$  # 2 ops, 2 holes, 1 arg

✗ def blackbox(x): return exp(x) + x     # 2 ops (should be 1)
✗ def blackbox(x, y): return (x+y)*(x-y) # 3 ops (should be 1)
✗ def blackbox(x): return x * 3.6        # has hole but hole not marked with $name$
✗ def blackbox(x): return sin(cos(x))    # 2 ops nested (should be 1)"""

    knobs = f"""
CURRENT COMPLEXITY TARGET: {json.dumps(complexity)}
YOU MUST generate a function matching EXACTLY these counts."""

    return f"{examples}\n{knobs}"


def complexity_adjustment_prompt(
    complexity: dict,
    rollouts_window: list,
    overall_mean: float,
    target_solve_rate: float,
) -> str:
    """prompt for first haiku call: determine new complexity"""
    return f"""recent functions and their mean rewards:
{json.dumps(rollouts_window[:10], indent=2)}

current complexity: {json.dumps(complexity)}
overall mean reward: {overall_mean:.2f}
target solve rate: {target_solve_rate}

adjust complexity (num_ops, num_holes, num_args) to move toward target solve rate.
if mean reward too low → decrease complexity
if mean reward too high → increase complexity

respond with json:
{complexity_json()}"""


def new_function_prompt(complexity: dict, rollouts_window: list) -> str:
    """prompt for second haiku call: generate function with given complexity"""
    recent_functions = (
        [r["blackbox"] for r in rollouts_window[:5]] if rollouts_window else []
    )

    return f"""generate a novel python function for reverse engineering game.

{complexity_explanation(complexity, step=1)}

recent functions (avoid duplicating):
{json.dumps(recent_functions, indent=2)}

respond with json using hypothesis strategies:
{spec_json()}"""


def oai_tools() -> list:
    """tool definitions for openai API"""
    return [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "execute bash command in persistent sandbox vm",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "bash command to execute",
                        },
                    },
                    "required": ["command"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "eval",
                "description": "execute python function with given kwargs in isolated environment",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "fn": {
                            "type": "string",
                            "description": "complete python function definition starting with 'def blackbox' including signature and return statement, NOT just an expression",
                        },
                        "kwargs": {
                            "type": "string",
                            "description": "JSON array of kwarg dicts to test function with",
                        },
                    },
                    "required": ["fn", "kwargs"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "submit",
                "description": "submit your predicted output",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "fn": {
                            "type": "string",
                            "description": "complete python function definition starting with 'def blackbox' including signature and return statement, NOT just an expression",
                        },
                        "output": {
                            "type": "number",
                            "description": "predicted output for the given input",
                        },
                    },
                    "required": ["fn", "output"],
                },
            },
        },
    ]
