#!/usr/bin/env python3
import click
from subprocess import run
from os import environ
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


@click.group()
def cli():
    pass


@cli.group()
def test():
    pass


@test.command()
def docker():
    """run tests with docker backend"""
    script_dir = Path(__file__).parent
    environ["SANDBOX_USE"] = "docker"
    run(
        ["uv", "run", "pytest", "sandbox_test.py"],
        cwd=script_dir,
        check=False,
    )


@test.command()
def wasm():
    """run tests with wasm backend"""
    script_dir = Path(__file__).parent
    environ["SANDBOX_USE"] = "wasm"
    run(
        ["uv", "run", "pytest", "sandbox_test.py"],
        cwd=script_dir,
        check=False,
    )


@cli.group()
def eval():
    pass


@eval.command()
def sonnet():
    """run evaluation with sonnet model"""
    script_dir = Path(__file__).parent
    environ["SANDBOX_USE"] = "python"
    run(
        [
            "uv",
            "run",
            "vf-eval",
            "unboxer",
            "-m",
            "anthropic/claude-sonnet-4.5",
            "--api-base-url",
            "https://openrouter.ai/api/v1",
            "-n",
            "1",
            "-a",
            '{"debug": true}',
        ],
        cwd=script_dir,
        check=False,
    )


@eval.command()
def haiku():
    """run evaluation with haiku model"""
    script_dir = Path(__file__).parent
    environ["SANDBOX_USE"] = "python"
    run(
        [
            "uv",
            "run",
            "vf-eval",
            "unboxer",
            "-m",
            "anthropic/claude-haiku-4.5",
            "--api-base-url",
            "https://openrouter.ai/api/v1",
            "-n",
            "1",
            "-a",
            '{"debug": true}',
        ],
        cwd=script_dir,
        check=False,
    )


def main():
    cli()


if __name__ == "__main__":
    main()
