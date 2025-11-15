#!/usr/bin/env bash

set -e

unset VIRTUAL_ENV

echo "→ installing main project..."
uv sync

echo "→ installing unboxer environment..."
uv pip install -e environments/unboxer

echo "→ testing sandbox..."
SANDBOX_USE=python uv run python -c "from sandbox import sandbox; print(sandbox('def f(x): return x*2', {'x': 21}).ok())"

echo ""
echo "✓ setup complete!"
echo ""

if [ ! -f .env ]; then
    echo "⚠ .env file not found"
    echo "create .env with:"
    echo "  OPENROUTER_API_KEY=your_key_here"
    echo ""
    echo "then run this script again"
    exit 1
fi

echo "→ activating venv and running haiku eval..."
source .venv/bin/activate
python un.py eval haiku
