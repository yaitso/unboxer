import asyncio
from environments.unboxer.unboxer import UnboxerEnv


async def test_generate_blackbox():
    env = UnboxerEnv(train_step=0, use_remote=False)

    blackbox_fn, kwargs_spec, holes_spec, io_pairs, eval_data = (
        await env._generate_blackbox_fn()
    )

    print("Generated function:")
    print(blackbox_fn)
    print()
    print(f"kwargs_spec: {kwargs_spec}")
    print(f"holes_spec: {holes_spec}")
    print(f"io_pairs: {io_pairs}")
    print(f"eval_data: {eval_data}")
    print()

    assert len(kwargs_spec) >= 1, "should have at least 1 argument"
    assert len(holes_spec) == 0, "train_step=0 should have 0 holes"
    assert len(io_pairs) >= 1, "should have at least 1 io pair"
    assert eval_data["expected_output"] is not None, "should have expected output"

    print("✓ all assertions passed")


if __name__ == "__main__":
    asyncio.run(test_generate_blackbox())
