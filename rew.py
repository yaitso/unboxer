#!/usr/bin/env python3
import click
from subprocess import run
from os import environ


@click.group()
def cli():
    pass


@cli.group()
def test():
    pass


@test.command()
def docker():
    """run tests with docker backend"""
    environ["SANDBOX_USE"] = "docker"
    run(["pytest", "sandbox_test.py"], check=False)


@test.command()
def wasm():
    """run tests with wasm backend"""
    environ["SANDBOX_USE"] = "wasm"
    run(["pytest", "sandbox_test.py"], check=False)


def main():
    cli()


if __name__ == "__main__":
    main()

