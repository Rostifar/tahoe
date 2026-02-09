#!/bin/bash
set -e

# RunPod deployment script
# Usage: ./deploy.sh <ssh-connection-string>
# Example: ./deploy.sh "ssh root@xyz.runpod.io -p 12345 -i ~/.ssh/id_rsa"

if [ -z "$1" ]; then
    echo "Usage: ./deploy.sh <ssh-connection-string>"
    echo "Example: ./deploy.sh \"ssh root@xyz.runpod.io -p 22345 -i ~/.ssh/id_rsa\""
    exit 1
fi

SSH_CMD="$1"
REMOTE_DIR="/workspace/tahoe"
LOCAL_DIR="$(dirname "$0")"

# Parse SSH command to extract host, port, and identity file for rsync
# Expected format: ssh root@host -p port -i keyfile
HOST=$(echo "$SSH_CMD" | grep -oE '[a-zA-Z0-9._-]+@[a-zA-Z0-9._-]+' | head -1)
PORT=$(echo "$SSH_CMD" | grep -oE '\-p [0-9]+' | awk '{print $2}')
KEY=$(echo "$SSH_CMD" | grep -oE '\-i [^ ]+' | awk '{print $2}')

if [ -z "$HOST" ]; then
    echo "Error: Could not parse host from SSH command"
    exit 1
fi

PORT=${PORT:-22}
SSH_OPTS="-p $PORT -o LogLevel=ERROR -o StrictHostKeyChecking=no"
[ -n "$KEY" ] && SSH_OPTS="$SSH_OPTS -i $KEY"

echo "Deploying to $HOST:$REMOTE_DIR"
echo ""

# Ensure GNU rsync is installed on remote (RunPod uses openrsync which is incompatible)
echo "==> Checking remote rsync..."
ssh -T $SSH_OPTS "$HOST" bash <<'REMOTE'
if ! rsync --version 2>/dev/null | grep -q "rsync  version"; then
    echo "Installing GNU rsync..."
    apt-get update -qq && apt-get install -y -qq rsync
fi
REMOTE

# Sync code and tokenizers
echo "==> Syncing code and tokenizers..."
rsync -avz --progress \
    --rsync-path=/usr/bin/rsync \
    --no-owner --no-group \
    --exclude '.venv' \
    --exclude '__pycache__' \
    --exclude '.git' \
    --exclude '.pytest_cache' \
    --exclude '.ruff_cache' \
    --exclude 'wandb' \
    --exclude 'checkpoints/*.pt' \
    --exclude '*.txt' \
    --exclude '.DS_Store' \
    --exclude 'notebooks' \
    --exclude 'historical' \
    --exclude 'tmp' \
    "$LOCAL_DIR/" \
    "$HOST:$REMOTE_DIR/" \
    -e "ssh -T $SSH_OPTS"

echo ""
echo "==> Sync complete!"
echo ""

# Ask if user wants to run setup
read -p "Run setup on remote (install dependencies)? [y/N] " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "==> Installing dependencies on remote..."
    ssh -T $SSH_OPTS "$HOST" bash <<REMOTE
cd $REMOTE_DIR
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="\$HOME/.local/bin:\$PATH"
fi
export PATH="\$HOME/.local/bin:\$PATH"
export UV_LINK_MODE=copy
uv sync
REMOTE
fi

echo ""
echo "==> Done! Connect with: $SSH_CMD"
echo "    Then: cd $REMOTE_DIR && python train.py"
