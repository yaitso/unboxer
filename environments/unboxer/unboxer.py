import json
from typing import Optional
from openai import AsyncOpenAI
import verifiers as vf
from verifiers.types import Messages, State
from sandbox import Sandbox
from db import RolloutsDB
import hypothesis.strategies as st
from datasets import Dataset


class UnboxerEnv(vf.MultiTurnEnv):
    def __init__(
        self,
        max_turns: int = 20,
        train_step: int = 0,
        rollouts_per_step: int = 1,
        use_remote: bool = True,
        train_run: Optional[int] = None,
        **kwargs,
    ):
        if "dataset" not in kwargs:
            kwargs["dataset"] = Dataset.from_dict(
                {"example_id": [0], "question": ["dummy"]}
            )
        super().__init__(max_turns=max_turns, **kwargs)
        self.max_turns = max_turns
        self.train_step = train_step
        self.rollouts_per_step = rollouts_per_step
        self.use_remote = use_remote
        self.train_run = train_run
        self.db = None
        self._db_initialized = False

    async def _ensure_db(self):
        if not self._db_initialized:
            self.db = RolloutsDB()
            await self.db.connect()
            self._db_initialized = True

    async def _generate_blackbox_fn(
        self, client: Optional[AsyncOpenAI] = None
    ) -> tuple[str, dict, dict, list, dict]:
        await self._ensure_db()

        if self.train_step == 0:
            complexity = {"num_ops": 1, "num_holes": 0, "num_args": 1}
            context_section = f"""this is the first train_step, so complexity settings are:
num_ops: {complexity["num_ops"]}
num_holes: {complexity["num_holes"]}
num_args: {complexity["num_args"]}

examples of valid functions:
- def blackbox(x): return sin(x)
- def blackbox(y): return y**y  # zero holes via arg reuse

NOT allowed:
- def blackbox(z): return z**2  # 2 is a hole of type <int>"""
        else:
            context = await self.db.get_adversarial_llm_context(window_size=100)
            overall_mean = (
                sum(r["mean_reward"] for r in context) / len(context)
                if context
                else 0.0
            )
            context_section = f"""recent unique functions and their mean rewards:
{json.dumps(context[:10], indent=2)}

overall mean reward: {overall_mean:.2f}
target solve rate: 0.4

adjust complexity (num_ops, num_holes, num_args) based on mean reward."""

        prompt = f"""generate a novel python function for a reverse engineering game.

COMPLEXITY KNOBS:
1. num_ops: count of operators + math functions (e.g., +, **, sin, cos)
   - only bool, int, float builtins and `from math import *` allowed
   - count_of_ops + count_of_functions = num_ops

2. num_holes: typed placeholders in the function
   - instead of `def blackbox(a): return a + 2`
   - use `def blackbox(a): return a + $c$`
   - holes let you generate function shapes, we sample values
   - mark holes with $name$ syntax

3. num_args: number of function arguments (self-explanatory)

{context_section}

respond with json using hypothesis strategies:
{{
    "fn": "def blackbox(a: float) -> float:\\n    return sin(a + $b$)",
    "kwargs": {{"a": "st.floats(-10, 10)"}},
    "holes": {{"b": "st.floats(0, 5)"}}
}}

note: all values will be rounded to 1 decimal place."""

        if client is None:
            from openai import AsyncOpenAI
            from os import environ

            client = AsyncOpenAI(
                api_key=environ.get("OPENAI_API_KEY"),
                base_url="https://openrouter.ai/api/v1",
            )

        response = await client.chat.completions.create(
            model="anthropic/claude-haiku-4.5",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
        )

        content = response.choices[0].message.content
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            data = json.loads(content[content.find("{") : content.rfind("}") + 1])

        blackbox_fn_template = data["fn"]
        kwargs_spec_str = data["kwargs"]
        holes_spec_str = data.get("holes", {})

        kwargs_spec = {}
        for param, strategy_str in kwargs_spec_str.items():
            kwargs_spec[param] = eval(strategy_str, {"st": st})

        holes_spec = {}
        for hole, strategy_str in holes_spec_str.items():
            holes_spec[hole] = eval(strategy_str, {"st": st})

        def sample_and_instantiate():
            sampled_holes = {}
            for hole, strategy in holes_spec.items():
                sampled_holes[hole] = round(strategy.example(), 1)

            instantiated_fn = blackbox_fn_template
            for hole, value in sampled_holes.items():
                instantiated_fn = instantiated_fn.replace(f"${hole}$", str(value))

            return instantiated_fn, sampled_holes

        blackbox_fn, sampled_holes = sample_and_instantiate()

        io_pairs = []
        for i in range(3):
            kwargs = {}
            for param, strategy in kwargs_spec.items():
                kwargs[param] = round(strategy.example(), 1)

            result = Sandbox.local(blackbox_fn, kwargs)
            if result.is_ok():
                output = round(result.ok()["output"], 1)
                io_pairs.append({"input": kwargs, "output": output})

        next_input = {}
        for param, strategy in kwargs_spec.items():
            next_input[param] = round(strategy.example(), 1)

        expected_result = Sandbox.local(blackbox_fn, next_input)
        expected_output = (
            round(expected_result.ok()["output"], 1)
            if expected_result.is_ok()
            else None
        )

        return (
            blackbox_fn,
            kwargs_spec,
            holes_spec,
            io_pairs,
            {"next_input": next_input, "expected_output": expected_output},
        )

    async def setup_state(self, state: State, **kwargs) -> State:
        client = kwargs.get("client")

        (
            blackbox_fn,
            kwargs_spec,
            holes_spec,
            io_pairs,
            next_data,
        ) = await self._generate_blackbox_fn(client)

        if self.use_remote:
            sandbox = Sandbox()
            machine_id = await sandbox.create()
            state["sandbox"] = sandbox
            state["machine_id"] = machine_id
        else:
            state["sandbox"] = None
            state["machine_id"] = "local"

        state["budget"] = self.max_turns
        state["io_pairs"] = io_pairs
        state["next_input"] = next_data["next_input"]
        state["blackbox_fn"] = blackbox_fn
        state["expected_output"] = next_data["expected_output"]

        await self._ensure_db()
        rollout_id = await self.db.add_rollout(
            train_step=self.train_step,
            rollout_name=state["machine_id"],
            blackbox=blackbox_fn,
            reward=0.0,
            train_run=self.train_run,
        )
        state["rollout_id"] = rollout_id

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

            if tool_name == "bash":
                args = json.loads(tool_call["function"]["arguments"])
                command = args["command"]

                if state["sandbox"]:
                    try:
                        output = await state["sandbox"].bash(command)
                    except Exception as e:
                        output = f"error: {str(e)}"
                else:
                    output = "error: bash tool only available with remote sandbox"

                tool_responses.append(
                    {
                        "role": "tool",
                        "content": output,
                        "tool_call_id": tool_call["id"],
                    }
                )

            elif tool_name == "eval":
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
                    result = Sandbox.local(fn, kw)
                    if result.is_ok():
                        output = round(result.ok()["output"], 1)
                        results.append({"input": kw, "output": output})
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

                expected = state["expected_output"]
                tolerance = 0.5

                if abs(predicted_output - expected) <= tolerance:
                    state["solved"] = True
                    reward = state["budget"]
                    await self.db.update_reward(state["rollout_id"], reward)

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
                            "content": f"✗ incorrect. expected: {expected}, got: {predicted_output} (tolerance: ±{tolerance}). try predicting output for: {state['next_input']}",
                            "tool_call_id": tool_call["id"],
                        }
                    )

            if state["budget"] <= 0:
                break

        if state.get("solved") or state["budget"] <= 0:
            if state["sandbox"]:
                await state["sandbox"].destroy()

        return tool_responses, state


def load_environment(debug: bool = False, **kwargs) -> vf.Environment:
    oai_tools = [
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

    rubric = vf.Rubric()

    def reward_func(completion, state):
        if state.get("solved", False):
            return float(state["budget"])
        return 0.0

    rubric.add_reward_func(reward_func)

    return UnboxerEnv(rubric=rubric, oai_tools=oai_tools, **kwargs)
