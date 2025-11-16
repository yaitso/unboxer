#!/usr/bin/env python3
import click
from subprocess import run
from pathlib import Path
from dotenv import load_dotenv
import re

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
def setup():
    """create modal volume for training (run once)"""
    result = run(
        ["modal", "volume", "list"],
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode == 0 and "unboxer-volume" in result.stdout:
        click.echo("volume 'unboxer-volume' already exists")
        return

    run(["modal", "volume", "create", "unboxer-volume", "--version=2"], check=True)
    click.echo("created volume 'unboxer-volume' (v2)")


@cli.command()
def train():
    """train unboxer model on modal h100"""
    script_dir = Path(__file__).parent

    run(["git", "add", "."], cwd=script_dir, check=True)
    run(["git", "commit", "-m", "yaitso"], cwd=script_dir, check=True)

    result = run(
        ["git", "rev-parse", "HEAD"],
        cwd=script_dir,
        capture_output=True,
        text=True,
        check=True,
    )
    commit_sha = result.stdout.strip()
    commit_6_chars = commit_sha[:6]

    env_file = script_dir / ".env"
    env_text = env_file.read_text()
    env_text_updated = re.sub(
        r"^TRAIN_COMMIT=.*$",
        f"TRAIN_COMMIT={commit_6_chars}",
        env_text,
        flags=re.MULTILINE,
    )

    if "TRAIN_COMMIT=" not in env_text:
        env_text_updated = env_text_updated.rstrip() + f"\nTRAIN_COMMIT={commit_6_chars}\n"

    env_file.write_text(env_text_updated)

    run(
        [
            "uv",
            "run",
            "modal",
            "run",
            str(
                script_dir / "train.py",
            ),
        ],
        cwd=script_dir,
        check=True,
    )


def main():
    cli()


if __name__ == "__main__":
    main()
