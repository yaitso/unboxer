FROM ghcr.io/astral-sh/uv:python3.12-trixie-slim

RUN apt-get update && apt-get install -y curl ca-certificates bzip2 && \
    curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y nodejs && \
    mkdir -p /opt/pyodide && \
    cd /opt/pyodide && \
    curl -L https://github.com/pyodide/pyodide/releases/download/0.29.0/pyodide-core-0.29.0.tar.bz2 | tar xj

COPY wasm.js ./wasm.js

ENTRYPOINT ["sh", "-c", "cat > /json && node wasm.js"]
