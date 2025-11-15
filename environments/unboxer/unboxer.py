import json
from datasets import Dataset
import verifiers as vf
from verifiers.types import Messages, State
from sandbox import sandbox


class UnboxerEnv(vf.MultiTurnEnv):
    def __init__(self, max_turns: int = 20, **kwargs):
        super().__init__(max_turns=max_turns, **kwargs)
        self.max_turns = max_turns

    async def setup_state(self, state: State, **kwargs) -> State:
        state["budget"] = self.max_turns
        state["io_pairs"] = state["info"]["io_pairs"]
        state["next_input"] = state["info"]["next_input"]
        state["blackbox_fn"] = state["info"]["blackbox_fn"]
        state["expected_output"] = state["info"]["expected_output"]
        return state

    async def is_completed(self, messages: Messages, state: State, **kwargs) -> bool:
        if state.get("solved", False):
            return True
        if state["budget"] <= 0:
            return True
        return await super().is_completed(messages, state, **kwargs)

    async def env_response(
        self, messages: Messages, state: State, **kwargs
    ) -> tuple[Messages, State]:
        assert isinstance(messages, list)
        last_msg = messages[-1]

        if "tool_calls" not in last_msg:
            return [], state

        tool_responses = []

        for tool_call in last_msg["tool_calls"]:
            state["budget"] -= 1
            tool_name = tool_call["function"]["name"]

            if tool_name == "sandbox":
                args = json.loads(tool_call["function"]["arguments"])
                fn = args["fn"]
                kwargs_str = args.get("kwargs", "[]")
                try:
                    kwargs_list = (
                        json.loads(kwargs_str)
                        if isinstance(kwargs_str, str)
                        else kwargs_str
                    )
                except json.JSONDecodeError:
                    kwargs_list = []

                results = []
                for kw in kwargs_list:
                    result = sandbox(fn, kw)
                    if result.is_ok():
                        results.append({"input": kw, "output": result.ok()})
                    else:
                        results.append({"input": kw, "error": result.err()})

                tool_responses.append(
                    {
                        "role": "tool",
                        "content": json.dumps({"results": results}),
                        "tool_call_id": tool_call["id"],
                    }
                )

            elif tool_name == "submit":
                args = json.loads(tool_call["function"]["arguments"])
                predicted_output = args["output"]

                if predicted_output == state["expected_output"]:
                    state["solved"] = True
                    reward = state["budget"]
                    tool_responses.append(
                        {
                            "role": "tool",
                            "content": f"✓ correct! reward: {reward}",
                            "tool_call_id": tool_call["id"],
                        }
                    )
                else:
                    state["next_input"] = state["info"].get(
                        "next_next_input", state["next_input"]
                    )
                    state["expected_output"] = state["info"].get(
                        "next_expected_output", state["expected_output"]
                    )

                    tool_responses.append(
                        {
                            "role": "tool",
                            "content": f"✗ incorrect. expected: {state['expected_output']}, got: {predicted_output}. try predicting output for: {state['next_input']}",
                            "tool_call_id": tool_call["id"],
                        }
                    )

            if state["budget"] <= 0:
                break

        return tool_responses, state


def load_environment(debug: bool = False, **kwargs) -> vf.Environment:
    oai_tools = [
        {
            "type": "function",
            "function": {
                "name": "sandbox",
                "description": "execute python function with given kwargs in sandboxed environment",
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
                "description": "submit your hypothesis function and predicted output",
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

    blackbox_fn = """
def blackbox(a: float, b: float):
    return a + (sin(a)*b)**2
"""

    result_1 = sandbox(blackbox_fn, {"a": 1.0, "b": 2.0})
    result_2 = sandbox(blackbox_fn, {"a": 2.0, "b": 3.0})
    result_3 = sandbox(blackbox_fn, {"a": 3.0, "b": 4.0})

    assert result_1.is_ok(), f"sandbox failed: {result_1.err()}"
    assert result_2.is_ok(), f"sandbox failed: {result_2.err()}"
    assert result_3.is_ok(), f"sandbox failed: {result_3.err()}"

    output_1 = result_1.ok()
    output_2 = result_2.ok()
    output_3 = result_3.ok()

    prompt_text = f"""this is an interactive reverse engineering game you'll be trained with RL.

at start of rollout you have budget of 20 turns. each time you call a tool it decrements by 1.
if budget reaches 0 episode terminates and you get 0 reward.
if you guess correctly before budget reaches 0 then reward = leftover budget.

given these input-output pairs:
- input: {{"a": 1.0, "b": 2.0}} → output: {output_1}
- input: {{"a": 2.0, "b": 3.0}} → output: {output_2}

your task: reverse engineer the blackbox function that produced them and predict output for:
input: {{"a": 3.0, "b": 4.0}}

you have 2 tools:

1. <sandbox>
   <fn>function to test</fn>
   <kwargs>[list of kwarg dicts to test]</kwargs>
   </sandbox>

2. <submit>
   <fn>your hypothesis function</fn>
   <output>predicted output for input {{"a": 3.0, "b": 4.0}}</output>
   </submit>
"""

    dummy_rows = [
        {
            "prompt": [{"role": "user", "content": prompt_text}],
            "info": {
                "io_pairs": [
                    {"input": {"a": 1.0, "b": 2.0}, "output": output_1},
                    {"input": {"a": 2.0, "b": 3.0}, "output": output_2},
                ],
                "next_input": {"a": 3.0, "b": 4.0},
                "blackbox_fn": blackbox_fn,
                "expected_output": output_3,
                "debug": debug,
            },
        }
    ]

    dataset = Dataset.from_list(dummy_rows)

    def reward_func(completion, state):
        if state.get("solved", False):
            return float(state["budget"])
        return 0.0

    rubric = vf.Rubric()
    rubric.add_reward_func(reward_func)

    return UnboxerEnv(
        dataset=dataset, rubric=rubric, max_turns=20, oai_tools=oai_tools, **kwargs
    )
