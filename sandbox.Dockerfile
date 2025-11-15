FROM ghcr.io/astral-sh/uv:python3.12-trixie-slim

COPY sandbox_run.py ./sandbox_run.py
COPY sandbox.toml ./pyproject.toml
RUN uv sync

ENTRYPOINT ["sh", "-c", "cat > /json && python3 sandbox_run.py"]
