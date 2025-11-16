import asyncio
import httpx
from pathlib import Path
from os import environ
from json import loads, dumps
from subprocess import run
from result import Result, Ok, Err
from typing import Optional
from uuid import uuid4
import xxhash


class Sandbox:
    def __init__(
        self,
        app_name: str = "unboxer",
        region: str = "iad",
        volume_size_gb: int = 1,
        memory_mb: int = 512,
        cpus: int = 1,
    ):
        self.app_name = app_name
        self.region = region
        self.volume_size_gb = volume_size_gb
        self.memory_mb = memory_mb
        self.cpus = cpus
        self.fly_api_token = environ.get("FLY_API_TOKEN")
        if not self.fly_api_token:
            raise ValueError("FLY_API_TOKEN not found in environment")

        self.machine_name = xxhash.xxh64(str(uuid4()).encode()).hexdigest()[:6]
        self.machine_id: Optional[str] = None
        self.volume_name: Optional[str] = None
        self.volume_id: Optional[str] = None
        self.base_url = "https://api.machines.dev/v1"
        self.headers = {
            "Authorization": f"Bearer {self.fly_api_token}",
            "Content-Type": "application/json",
        }

    async def create(self) -> str:
        async with httpx.AsyncClient() as client:
            self.volume_name = f"vol_{self.machine_name}"
            volume_payload = {
                "name": self.volume_name,
                "size_gb": self.volume_size_gb,
                "region": self.region,
            }

            vol_resp = await client.post(
                f"{self.base_url}/apps/{self.app_name}/volumes",
                headers=self.headers,
                json=volume_payload,
            )
            vol_resp.raise_for_status()
            vol_data = vol_resp.json()
            self.volume_id = vol_data["id"]

            machine_payload = {
                "name": f"sandbox_{self.machine_name}",
                "region": self.region,
                "config": {
                    "image": f"registry.fly.io/{self.app_name}:latest",
                    "services": [
                        {
                            "ports": [{"port": 22}],
                            "protocol": "tcp",
                            "internal_port": 2222,
                        }
                    ],
                    "mounts": [
                        {
                            "volume": self.volume_id,
                            "path": "/workspace",
                        }
                    ],
                    "guest": {
                        "cpu_kind": "shared",
                        "cpus": self.cpus,
                        "memory_mb": self.memory_mb,
                    },
                },
            }

            machine_resp = await client.post(
                f"{self.base_url}/apps/{self.app_name}/machines",
                headers=self.headers,
                json=machine_payload,
            )
            machine_resp.raise_for_status()
            machine_data = machine_resp.json()
            self.machine_id = machine_data["id"]

            await self._wait_for_machine()

            return self.machine_id

    async def _wait_for_machine(self, timeout: int = 120):
        start = asyncio.get_event_loop().time()
        async with httpx.AsyncClient() as client:
            while True:
                if asyncio.get_event_loop().time() - start > timeout:
                    raise TimeoutError(
                        f"machine {self.machine_id} did not start in {timeout}s"
                    )

                resp = await client.get(
                    f"{self.base_url}/apps/{self.app_name}/machines/{self.machine_id}",
                    headers=self.headers,
                )
                resp.raise_for_status()
                data = resp.json()

                if data.get("state") == "started":
                    break

                await asyncio.sleep(2)

    async def bash(self, command: str) -> str:
        if not self.machine_id:
            raise ValueError("machine not created yet")

        import shlex
        
        shell_cmd = f"sh -c {shlex.quote(command)}"
        full_command = f"flyctl machine exec -a {self.app_name} {self.machine_id} {shlex.quote(shell_cmd)}"

        proc = await asyncio.create_subprocess_shell(
            full_command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await proc.communicate()

        stdout_str = stdout.decode().strip()
        stderr_str = stderr.decode().strip()

        combined = stdout_str
        if stderr_str:
            if combined:
                combined = f"{combined}\nstderr:\n{stderr_str}"
            else:
                combined = f"stderr:\n{stderr_str}"

        return combined or "(no output)"

    async def destroy(self):
        async with httpx.AsyncClient() as client:
            if self.machine_id:
                try:
                    await client.delete(
                        f"{self.base_url}/apps/{self.app_name}/machines/{self.machine_id}",
                        headers=self.headers,
                    )
                except httpx.HTTPStatusError:
                    pass

            if self.volume_id:
                try:
                    await client.delete(
                        f"{self.base_url}/apps/{self.app_name}/volumes/{self.volume_id}",
                        headers=self.headers,
                    )
                except httpx.HTTPStatusError:
                    pass

        self.machine_id = None
        self.volume_id = None

    @staticmethod
    def local(fn: str, kwargs: dict) -> Result[dict, str]:
        script_dir = Path(__file__).parent
        sandbox_run_code = (script_dir / "sandbox.template.py").read_text()

        payload = dumps({"fn": fn, "kwargs": kwargs})

        try:
            result = run(
                ["python3", "-c", sandbox_run_code],
                input=payload,
                capture_output=True,
                text=True,
                timeout=15,
            )
        except Exception as e:
            return Err(f"python execution failed: {str(e)}")

        try:
            data = loads(result.stdout)
            if "error" in data:
                return Err(data["error"])
            return Ok({"output": data["result"]})
        except Exception:
            return Err(
                f"json decode failed - stderr: {result.stderr}, stdout: {result.stdout}"
            )
