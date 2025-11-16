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
examples of valid functions:
- def blackbox(x): return sin(x)  # 1 op, 0 holes, 1 arg
- def blackbox(y): return y**y    # 1 op, 0 holes, 1 arg (arg reuse)

NOT allowed:
- def blackbox(z): return z**2    # 2 is a constant hole"""

    knobs = f"""COMPLEXITY KNOBS:
1. num_ops: count of operators + math functions (e.g., +, **, sin, cos)
2. num_holes: typed placeholders marked with $name$ syntax
3. num_args: number of function arguments

current complexity: {json.dumps(complexity)}"""

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
                            "description": "python function definition to execute",
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
                            "description": "your hypothesis function definition",
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
