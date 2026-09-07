# 💻 Commands Reference

## 👨‍💻 For Users

### 1. Docker Host Commands
```bash
# Launch Tensorbox (no inline keys needed)
docker run -d --name tensorbox -p 8080:8080 -p 8888:8888 -p 8501:8501 -p 5000:5000 -p 6006:6006 -v "${PWD}/workspace:/workspace" tensorbox:latest

# Stop Tensorbox
docker stop tensorbox

# Resume a stopped Tensorbox
docker start tensorbox

# View container logs from host
docker logs tensorbox
```

### 2. In-Container `tensorbox` CLI Commands (Inside VS Code Terminal)
```bash
# Configuration & Diagnostics
tensorbox set-key openai sk-proj-...     # Set OpenAI API key in .env
tensorbox set-key anthropic sk-ant-...    # Set Anthropic API key
tensorbox set-key google your-google-key  # Set Gemini API key
tensorbox config                          # Interactive configuration wizard
tensorbox status                          # Check health of all services & keys
tensorbox logs [service]                  # View live logs (jupyter, code-server, tensorbox, postgres)
tensorbox bug-report                      # Create zipped diagnostics bundle for instructors/TAs

# Project Scaffolding & Git Submission
tensorbox new <project-name>              # Create starter project with main.py, README, and tests
tensorbox git-init                        # Initialize secure user Git repo (auto-ignores .env/data/models)

# Service Launchers
tensorbox fastapi [file.py]               # Launch FastAPI with OpenAPI Swagger UI (:5000/docs)
tensorbox streamlit [file.py]             # Launch Streamlit dashboard (:8501)
tensorbox chroma                          # Launch local ChromaDB vector store (:8000)
tensorbox tensorboard [logdir]            # Launch TensorBoard visualizer (:6006)
tensorbox postgres                        # Start & verify PostgreSQL (:5432)
tensorbox sqlite                          # View SQLite database path & connection code
tensorbox mysql                           # View MySQL / MariaDB connection guidance
tensorbox redis                           # Start & verify Redis (:6379)
tensorbox test                            # Run test suite
```

> [!NOTE]
> `devbox` works as a full alias for `tensorbox` (e.g., `devbox status` = `tensorbox status`).

---

## 👨‍💻 For Developers & Instructors

```bash
# Run unit tests locally
pytest tests

# Automated dependency solver & lockfile compiler
python scripts/resolve_dependencies.py   # Or: tensorbox resolve-deps

# Automated dependency audit & library smoke test
python scripts/check_dependencies.py     # Or: tensorbox check-deps

# Build image locally
docker build -t tensorbox:latest .

# Multi-architecture build & push for Docker Hub Marketplace
docker buildx build --platform linux/amd64,linux/arm64 -t <your-username>/tensorbox:latest -t <your-username>/tensorbox:1.0.0 --push .
```
