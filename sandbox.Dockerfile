FROM --platform=linux/amd64 rust:1.91-slim-trixie AS builder
WORKDIR /usr/src/proxy
COPY ./sandbox/proxy .
RUN cargo install --path .

FROM --platform=linux/amd64 ghcr.io/astral-sh/uv:python3.12-trixie-slim

RUN apt update && apt install -y --no-install-recommends \
    openssh-server \
    sudo \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -ms /bin/bash agent && \
    usermod -aG sudo agent && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

RUN mkdir -p /run/sshd && \
    sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config && \
    echo "AuthorizedKeysFile /etc/ssh/authorized_keys/%u" >> /etc/ssh/sshd_config

RUN mkdir -p /etc/ssh/authorized_keys

COPY unboxer_ssh.pub /etc/ssh/authorized_keys/agent
RUN chmod 644 /etc/ssh/authorized_keys/agent && \
    chown root:root /etc/ssh/authorized_keys/agent

COPY --from=builder /usr/local/cargo/bin/proxy /usr/local/bin/proxy

USER agent
WORKDIR /workspace

CMD ["bash", "-c", "sudo /usr/sbin/sshd && exec proxy"]
