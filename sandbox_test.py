#!/usr/bin/env python3
import pytest
from dotenv import load_dotenv
from sandbox import Sandbox

load_dotenv()


@pytest.mark.asyncio
async def test_remote_sandbox_persistence():
    """test that bash commands persist across multiple calls via SSH"""
    sandbox = Sandbox()

    try:
        machine_id = await sandbox.create()

        test_content = f"persistence_test_{sandbox.machine_name}"
        await sandbox.bash(f"echo '{test_content}' > /workspace/test.txt")

        read_result = await sandbox.bash("cat /workspace/test.txt")

        assert test_content in read_result, (
            f"expected '{test_content}' in output, got: {read_result}"
        )

        ls_result = await sandbox.bash("ls -la /workspace")
        assert "test.txt" in ls_result

    finally:
        await sandbox.destroy()


@pytest.mark.asyncio
async def test_sandbox_local():
    """test local python execution via Sandbox.local()"""
    fn = """
def blackbox(a: float, b: float) -> float:
    from math import sin
    return a + (sin(a) * b) ** 2
"""

    result = Sandbox.local(fn, {"a": 1.0, "b": 2.0})
    assert result.is_ok()

    output = result.ok()["output"]

    from math import sin

    expected = 1.0 + (sin(1.0) * 2.0) ** 2
    assert abs(output - expected) < 1e-10
