#!/usr/bin/env python3
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib

import verifiers as vf


def main():
    config_path = Path("configs/unboxer.toml")
    if not config_path.exists():
        raise SystemExit(f"config not found: {config_path}")

    with config_path.open("rb") as f:
        config = tomllib.load(f)

    model = config["model"]
    env_id = config["env"]["id"]
    env_args = config["env"].get("args", {})

    env = vf.load_environment(env_id=env_id, **env_args)
    rl_config = vf.RLConfig(**config["trainer"].get("args", {}))
    trainer = vf.RLTrainer(model=model, env=env, args=rl_config)
    trainer.train()


if __name__ == "__main__":
    main()
