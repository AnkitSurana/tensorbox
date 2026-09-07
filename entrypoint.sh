#!/bin/bash
set -e

# 1. Ensure workspace directories exist
mkdir -p /workspace/projects /workspace/notebooks /workspace/data /workspace/models /workspace/logs

# 2. Automatically create .env in /workspace if not present
if [ ! -f /workspace/.env ]; then
    if [ -f /opt/tensorbox/.env.example ]; then
        cp /opt/tensorbox/.env.example /workspace/.env
    elif [ -f /opt/devbox/.env.example ]; then
        cp /opt/devbox/.env.example /workspace/.env
    else
        touch /workspace/.env
    fi
fi

# 3. Automatically create secure .gitignore in /workspace to prevent accidental secret/data leakage
if [ ! -f /workspace/.gitignore ]; then
    cat <<'EOF' > /workspace/.gitignore
# Environment & Secrets (NEVER COMMIT)
.env
.env.*
*.key
*.pem
secrets/
credentials/

# Heavy Data, Datasets & Vector Stores
data/
datasets/
*_data/
chroma_data/
qdrant_data/
weaviate_data/
milvus_data/
postgres_data/
redis_data/
*.csv
*.tsv
*.parquet
*.h5
*.hdf5
*.sqlite
*.sqlite3
*.db

# Heavy Neural Weights & Checkpoints
models/
*.pt
*.pth
*.bin
*.onnx
*.safetensors
*.ckpt
runs/
wandb/

# Runtime Logs & Diagnostics
logs/
*.log
*.zip

# Python & IDE Cache
__pycache__/
*.py[cod]
*$py.class
.ipynb_checkpoints/
.pytest_cache/
.coverage
htmlcov/
.venv/
venv/
env/
.DS_Store
Thumbs.db
EOF
fi

# 3. Write container diagnostic info to /workspace/logs/system_info.log
cat <<EOF > /workspace/logs/system_info.log
Tensorbox Container Diagnostic Information
============================================
Boot Timestamp: $(date -u +"%Y-%m-%d %H:%M:%S UTC")
Python Version: $(python3 --version 2>&1)
Memory Info:    $(free -h 2>/dev/null || cat /proc/meminfo | grep MemTotal || echo "N/A")
Disk Info:      $(df -h /workspace | tail -1)
OS Architecture: $(uname -a)
Workspace Path: /workspace
============================================
EOF

# 4. Check if user called a sub-command (e.g. tensorbox status, tensorbox set-key, etc.)
if [ "$1" != "" ] && [ "$1" != "daemon" ]; then
    if [ "$1" = "bash" ] || [ "$1" = "sh" ]; then
        exec /bin/bash "${@:2}"
    else
        exec python3 -m utils.cli "$@"
    fi
fi

# 5. Default daemon mode: Start VS Code (code-server) and JupyterLab
echo "=========================================================="
echo "🚀 Starting Tensorbox Workspace..."
echo "=========================================================="

# Start PostgreSQL & Redis services
echo "🐘 Initializing PostgreSQL database service..."
mkdir -p /var/run/postgresql && chown -R postgres:postgres /var/run/postgresql
service postgresql start >/dev/null 2>&1 || true
su - postgres -c "psql -c \"ALTER USER postgres WITH PASSWORD 'postgres';\"" >/dev/null 2>&1 || true
su - postgres -c "psql -c \"CREATE DATABASE ml_database;\"" >/dev/null 2>&1 || true

echo "⚡ Initializing Redis cache service..."
service redis-server start >/dev/null 2>&1 || true

# Start code-server (VS Code in Browser) on port 8080
if command -v code-server >/dev/null 2>&1; then
    echo "💻 Launching VS Code Server on http://0.0.0.0:8080 ..."
    code-server \
        --bind-addr 0.0.0.0:8080 \
        --auth none \
        --disable-telemetry \
        /workspace > /workspace/logs/code-server.log 2>&1 &
fi

# Start JupyterLab on port 8888
echo "📓 Launching JupyterLab on http://0.0.0.0:8888 ..."
jupyter lab \
    --ip=0.0.0.0 \
    --port=8888 \
    --no-browser \
    --allow-root \
    --ServerApp.token='' \
    --ServerApp.password='' \
    --ServerApp.root_dir=/workspace > /workspace/logs/jupyter.log 2>&1 &

echo ""
echo "=========================================================="
echo "✅ Tensorbox is online and ready!"
echo "👉 VS Code Browser: http://localhost:8080"
echo "👉 JupyterLab:       http://localhost:8888"
echo "👉 Logs directory:   /workspace/logs (saved on host machine)"
echo "👉 CLI Utility:      run 'tensorbox help' in terminal"
echo "=========================================================="
echo ""

# Stream background logs to container output
tail -f /workspace/logs/code-server.log /workspace/logs/jupyter.log 2>/dev/null

