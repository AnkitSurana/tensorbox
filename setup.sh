#!/bin/bash
set -e

echo "🚀 Setting up Tensorbox Environment..."

# Detect docker compose vs docker-compose
if docker compose version >/dev/null 2>&1; then
    COMPOSE="docker compose"
elif docker-compose version >/dev/null 2>&1; then
    COMPOSE="docker-compose"
else
    echo "❌ Error: Docker / Docker Compose is not installed or not running."
    echo "👉 Install Docker Desktop: https://www.docker.com/products/docker-desktop"
    exit 1
fi

# Create directories
mkdir -p notebooks data datasets projects models utils tests logs

touch notebooks/.gitkeep data/.gitkeep datasets/.gitkeep projects/.gitkeep models/.gitkeep
echo "✅ Workspace directories initialized"

# Setup .env
if [ ! -f .env ]; then
    cp .env.example .env
    echo "⚙️  Created .env from template"
fi

# Build
echo "📦 Building Tensorbox Docker image..."
$COMPOSE build

# Start
echo "🎬 Starting services..."
$COMPOSE up -d

echo ""
echo "============================================================"
echo "✅ Setup complete!"
echo "📖 Access Services:"
echo "   • VS Code (Browser): http://localhost:8080"
echo "   • JupyterLab:        http://localhost:8888"
echo "   • ChromaDB:          http://localhost:8000"
echo "   • FastAPI / OpenAPI: http://localhost:5000/docs"
echo "============================================================"
echo ""

