#!/usr/bin/env bash

set -e

echo "→ installing main project..."
uv sync

echo "→ installing unboxer environment..."
uv pip install -e environments/unboxer

echo "✓ setup complete. run 'uv run un eval haiku' to test"
