#!/usr/bin/env python3
import click
from subprocess import run
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


@click.group()
def cli():
    pass


@cli.command()
def build():
    """build and push unboxer sandbox image to fly.io"""
    script_dir = Path(__file__).parent
    build_script = script_dir / "sandbox.sh"

    run(["bash", str(build_script)], cwd=script_dir, check=True)


@cli.group()
def eval():
    """run evaluations"""
    pass


@eval.command()
def haiku():
    """run evaluation with claude haiku"""
    script_dir = Path(__file__).parent
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
        ],
        cwd=script_dir,
        check=False,
    )


@eval.command()
def sonnet():
    """run evaluation with claude sonnet"""
    script_dir = Path(__file__).parent
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
        ],
        cwd=script_dir,
        check=False,
    )


@cli.command()
def train():
    """train unboxer model (not implemented yet)"""
    click.echo("training not implemented yet")


def main():
    cli()


if __name__ == "__main__":
    main()
