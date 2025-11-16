import json
from typing import Optional
from openai import AsyncOpenAI
import verifiers as vf
from verifiers.types import Messages, State
from sandbox import Sandbox
from db import RolloutsDB
import hypothesis.strategies as st
import prompts


def sample_holes(holes_spec: dict) -> dict:
    """sample values for holes from hypothesis strategies"""
    sampled = {}
    for hole, strategy in holes_spec.items():
        sampled[hole] = round(strategy.example(), 1)
    return sampled


def instantiate_function(template: str, holes: dict) -> str:
    """replace $hole$ placeholders with sampled values"""
    fn = template
    for hole, value in holes.items():
        fn = fn.replace(f"${hole}$", str(value))
    return fn


def sample_kwargs(kwargs_spec: dict) -> dict:
    """sample kwargs from hypothesis strategies"""
    return {k: round(v.example(), 1) for k, v in kwargs_spec.items()}


class UnboxerEnv(vf.MultiTurnEnv):
    def __init__(
        self,
        max_turns: int = 20,
        train_step: int = 0,
        rollouts_per_step: int = 1,
        use_remote: bool = True,
        train_run: Optional[int] = None,
        target_solve_rate: float = 0.4,
        dsn: Optional[str] = None,
        train_commit: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(max_turns=max_turns, **kwargs)
        from os import environ

        self.max_turns = max_turns
        self.train_step = train_step
        self.rollouts_per_step = rollouts_per_step
        self.use_remote = use_remote
        self.train_run = train_run
        self.target_solve_rate = target_solve_rate
        self.dsn = dsn or environ.get("POSTGRES")
        self.train_commit = train_commit or environ.get("TRAIN_COMMIT")
        self.db: RolloutsDB = None  # type: ignore
        self.db_initialized = False
        self.current_complexity = {"num_ops": 1, "num_holes": 0, "num_args": 1}

    async def ensure_db(self):
        if not self.db_initialized:
            self.db = RolloutsDB(dsn=self.dsn)
            await self.db.connect()
            if self.train_run is None:
                self.train_run = await self.db.get_next_train_run()
            self.db_initialized = True

    async def get_or_create_client(
        self, client: Optional[AsyncOpenAI] = None
    ) -> AsyncOpenAI:
        """get existing client or create new one"""
        if client is not None:
            return client

        from openai import AsyncOpenAI
        from os import environ

        return AsyncOpenAI(
            api_key=environ.get("OPENAI_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
        )

    async def adjust_complexity(self, client: Optional[AsyncOpenAI] = None) -> dict:
        """first haiku call: determine new complexity based on db stats"""
        await self.ensure_db()
        client = await self.get_or_create_client(client)

        rollout_window = await self.db.get_rollout_window(window_size=100)
        overall_mean = (
            sum(r["mean_reward"] for r in rollout_window) / len(rollout_window)
            if rollout_window
            else 0.0
        )

        prompt = prompts.complexity_adjustment_prompt(
            self.current_complexity,
            rollout_window,
            overall_mean,
            self.target_solve_rate,
        )

        response = await client.chat.completions.create(
            model="anthropic/claude-haiku-4.5",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
        )

        content = response.choices[0].message.content
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            data = json.loads(content[content.find("{") : content.rfind("}") + 1])

        return data

    def validate_complexity(self, fn: str, complexity: dict) -> bool:
        """validate that generated function matches complexity constraints"""
        import re

        ops = [
            "+",
            "-",
            "*",
            "/",
            "**",
            "sin",
            "cos",
            "tan",
            "exp",
            "log",
            "sqrt",
            "abs",
        ]

        op_count = 0
        for op in ops:
            if op in ["**"]:
                op_count += fn.count(op)
            else:
                op_count += len(re.findall(rf"\b{re.escape(op)}\b", fn))

        hole_count = len(re.findall(r"\$\w+\$", fn))

        arg_match = re.search(r"def blackbox\((.*?)\)", fn)
        arg_count = len(arg_match.group(1).split(",")) if arg_match else 0

        return (
            op_count == complexity["num_ops"]
            and hole_count == complexity["num_holes"]
            and arg_count == complexity["num_args"]
        )

    async def generate_blackbox_fn(
        self, complexity: dict, client: Optional[AsyncOpenAI] = None
    ) -> tuple[str, dict, dict, list, dict]:
        """second haiku call: generate function with given complexity"""
        await self.ensure_db()
        client = await self.get_or_create_client(client)

        rollout_window = await self.db.get_rollout_window(window_size=100)

        for attempt in range(3):
            prompt = prompts.new_function_prompt(complexity, rollout_window)

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

            if self.validate_complexity(blackbox_fn_template, complexity):
                break
            elif attempt == 2:
                pass

        kwargs_spec_str = data["kwargs"]
        holes_spec_str = data.get("holes", {})

        kwargs_spec = {
            param: eval(strategy_str, {"st": st})
            for param, strategy_str in kwargs_spec_str.items()
        }
        holes_spec = {
            hole: eval(strategy_str, {"st": st})
            for hole, strategy_str in holes_spec_str.items()
        }

        sampled_holes = sample_holes(holes_spec)
        blackbox_fn = instantiate_function(blackbox_fn_template, sampled_holes)

        n_input_output_pairs = []
        for _ in range(3):
            kwargs = sample_kwargs(kwargs_spec)
            result = Sandbox.local(blackbox_fn, kwargs)
            if result.is_ok():
                output = round(result.ok()["output"], 1)
                n_input_output_pairs.append({"input": kwargs, "output": output})

        n_plus_one_input = sample_kwargs(kwargs_spec)
        expected_result = Sandbox.local(blackbox_fn, n_plus_one_input)
        n_plus_one_output = (
            round(expected_result.ok()["output"], 1)
            if expected_result.is_ok()
            else None
        )

        return (
            blackbox_fn,
            kwargs_spec,
            holes_spec,
            n_input_output_pairs,
            {
                "n_plus_one_input": n_plus_one_input,
                "n_plus_one_output": n_plus_one_output,
            },
        )

    async def setup_state(self, state: State, **kwargs) -> State:
        client = kwargs.get("client")

        await self.ensure_db()

        if self.train_step == 0:
            complexity = {"num_ops": 1, "num_holes": 0, "num_args": 1}
        else:
            complexity = await self.adjust_complexity(client)
            self.current_complexity = complexity

        (
            blackbox_fn,
            kwargs_spec,
            holes_spec,
            n_input_output_pairs,
            next_data,
        ) = await self.generate_blackbox_fn(complexity, client)

        if self.use_remote:
            sandbox = Sandbox()
            machine_id = await sandbox.create()
            state["sandbox"] = sandbox
            state["machine_id"] = machine_id
        else:
            state["sandbox"] = None
            state["machine_id"] = "local"

        state["budget"] = self.max_turns
        state["complexity"] = complexity
        state["holes_spec"] = holes_spec
        state["n_input_output_pairs"] = n_input_output_pairs
        state["n_plus_one_input"] = next_data["n_plus_one_input"]
        state["blackbox_fn"] = blackbox_fn
        state["n_plus_one_output"] = next_data["n_plus_one_output"]
        state["kwargs_spec"] = kwargs_spec

        rollout_id = await self.db.add_rollout(
            train_step=self.train_step,
            rollout_name=state["machine_id"],
            blackbox=blackbox_fn,
            reward=0.0,
            train_run=self.train_run,
            train_commit=self.train_commit,
        )
        state["rollout_id"] = rollout_id

        game_prompt = f"""this is an interactive reverse engineering game playing which you will be trained with RL
at the start of a rollout you have budget of {self.max_turns} turns
each time you call tool it is decremented by 1
if it reaches 0 episode is terminated and you get 0 reward
if you guess correctly before your budget reaches 0 then your reward is leftover budget
so keep track of budget you have left to ensure you will be able to call <submit>

if you fail to predict proper output we tell you so in tool response, provide actual output value, and give new input to predict, and we repeat that until rollout ends

given the list of {len(n_input_output_pairs)} input-output pairs: {n_input_output_pairs}
your task is to reverse engineer blackbox function that produced them and predict output for next input: {next_data["n_plus_one_input"]}

you have access to 3 tools:
1. bash: execute bash command in persistent sandbox vm (only if remote sandbox enabled)
2. eval: execute python function with given kwargs in isolated environment
3. submit: submit your predicted output along with hypothesis function

IMPORTANT: when calling eval or submit, the 'fn' parameter MUST be a complete python function definition like:
def blackbox(x: float) -> float:
    return x + 2

NOT just an expression like "x + 2" — that will fail to execute"""

        state["prompt"] = [{"role": "user", "content": game_prompt}]

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

                expected = state["n_plus_one_output"]
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
                    n_plus_one_input = sample_kwargs(state["kwargs_spec"])
                    expected_result = Sandbox.local(
                        state["blackbox_fn"], n_plus_one_input
                    )
                    n_plus_one_output = (
                        round(expected_result.ok()["output"], 1)
                        if expected_result.is_ok()
                        else None
                    )

                    state["n_plus_one_input"] = n_plus_one_input
                    state["n_plus_one_output"] = n_plus_one_output

                    tool_responses.append(
                        {
                            "role": "tool",
                            "content": f"✗ incorrect. expected: {expected}, got: {predicted_output} (tolerance: ±{tolerance}). try predicting output for: {state['n_plus_one_input']}",
                            "tool_call_id": tool_call["id"],
                        }
                    )

            if state["budget"] <= 0:
                break

        if state.get("solved") or state["budget"] <= 0:
            await self.db.update_trajectory(
                rollout_id=state["rollout_id"],
                trajectory=state["completion"],
                num_turns=state.get("turn", 0),
                solved=state.get("solved", False),
            )

            if state["sandbox"]:
                await state["sandbox"].destroy()

        return tool_responses, state


def load_environment(
    debug: bool = False, num_examples: int = 100, **kwargs
) -> vf.Environment:
    if "dataset" not in kwargs:
        from datasets import Dataset

        kwargs["dataset"] = Dataset.from_dict(
            {
                "example_id": list(range(num_examples)),
                "question": ["dummy"] * num_examples,
            }
        )

    rubric = vf.Rubric()

    def reward_func(completion, state):
        if state.get("solved", False):
            return float(state["budget"])
        return 0.0

    rubric.add_reward_func(reward_func)

    return UnboxerEnv(rubric=rubric, oai_tools=prompts.oai_tools(), **kwargs)
