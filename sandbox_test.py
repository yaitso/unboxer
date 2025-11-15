import pytest
from sandbox import sandbox, build


@pytest.fixture(scope="session", autouse=True)
def build_docker_image():
    build()
    yield


class TestSandboxBasic:
    def test_simple_arithmetic(self):
        result = sandbox("def blackbox(x): return x * 2", {"x": 21})
        assert result.is_ok()
        assert result.ok() == {"result": 42}

    def test_math_functions(self):
        result = sandbox("def blackbox(x): return sin(x) + cos(x)", {"x": 0})
        assert result.is_ok()
        data = result.ok()
        assert "result" in data
        assert abs(data["result"] - 1.0) < 1e-10

    def test_list_comprehension(self):
        result = sandbox(
            "def blackbox(nums): return [x**2 for x in nums]",
            {"nums": [1, 2, 3, 4, 5]},
        )
        assert result.is_ok()
        assert result.ok() == {"result": [1, 4, 9, 16, 25]}

    def test_snake(self):
        from math import sin

        result = sandbox(
            "def blackbox(a: float, b: float): return a + (sin(a)*b)**2",
            {"a": 1.0, "b": 2.0},
        )
        assert result.is_ok()
        expected = 1.0 + (sin(1.0) * 2.0) ** 2
        assert abs(result.ok()["result"] - expected) < 1e-10

    def test_missing_blackbox_function(self):
        result = sandbox("def foo(): return 42", {})
        assert result.is_ok()
        data = result.ok()
        assert "error" in data
        assert "blackbox" in data["error"]

    def test_exception_handling(self):
        result = sandbox("def blackbox(): return 1/0", {})
        assert result.is_ok()
        data = result.ok()
        assert "error" in data
        assert "division" in data["error"].lower()


class TestSandboxResourceLimits:
    def test_infinite_loop_timeout(self):
        result = sandbox(
            """
def blackbox():
    while True:
        pass
""",
            {},
        )
        assert result.is_err()

    def test_memory_bomb(self):
        result = sandbox(
            """
def blackbox():
    x = [0] * (10**9)
    return len(x)
""",
            {},
        )
        assert result.is_err() or ("error" in result.ok() if result.is_ok() else False)

    def test_recursion_limit(self):
        result = sandbox(
            """
def blackbox():
    def recurse():
        return recurse()
    return recurse()
""",
            {},
        )
        assert result.is_err()
        assert "timed out" in result.err()


class TestSandboxFileSystem:
    def test_tmpfs_write(self):
        result = sandbox(
            """
def blackbox():
    with open('/tmp/test.txt', 'w') as f:
        f.write('hello world')
    return 'success'
""",
            {},
        )
        assert result.is_ok()
        assert result.ok() == {"result": "success"}

    def test_tmpfs_read_write(self):
        result = sandbox(
            """
def blackbox():
    with open('/tmp/test2.txt', 'w') as f:
        f.write('data')
    with open('/tmp/test2.txt', 'r') as f:
        return f.read()
""",
            {},
        )
        assert result.is_ok()
        assert result.ok() == {"result": "data"}

    def test_file_size_limit(self):
        result = sandbox(
            """
def blackbox():
    with open('/tmp/large.txt', 'w') as f:
        f.write('x' * (2 * 1024 * 1024))
    return 'success'
""",
            {},
        )
        assert result.is_ok()
        assert "error" in result.ok()
