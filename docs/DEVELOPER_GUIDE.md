# 🛠️ Tensorbox Developer & Instructor Guide

This repository contains the source definitions, test suites, utilities, and build recipes used to maintain and publish the `tensorbox` Docker image for users.

---

## 🏗️ Architecture

The image is built on Python 3.11 with `code-server` (VS Code in the browser), JupyterLab, pre-installed NLP models, embedded databases, and 200+ ML packages.

```text
Host System (Mac / Windows / Linux)
  └── docker run -p 8080:8080 -p 8888:8888 -p 8501:8501 -p 5000:5000 -v "${PWD}/workspace:/workspace"
        │
        └── Container: tensorbox
              ├── code-server (VS Code web IDE on :8080)
              ├── JupyterLab (Interactive Notebooks on :8888)
              ├── FastAPI / OpenAPI (Swagger docs on :5000/docs)
              ├── Streamlit (Dashboard UI on :8501)
              ├── PostgreSQL, Redis, ChromaDB (Local Daemons)
              ├── /workspace/logs (Host-Visible Persistent Logs)
              └── /opt/tensorbox/utils (Baked-in Config & Health Checks)
```

---

## 📋 1. Host-Visible Logging & Bug Diagnostics

All services write logs directly to the mounted directory (`/workspace/logs`), which is immediately accessible in the `workspace/logs` folder on the user's physical host machine:

| Log File | Purpose |
| :--- | :--- |
| `workspace/logs/tensorbox.log` | Unified Python & CLI activity logs |
| `workspace/logs/code-server.log` | VS Code web server and extension output |
| `workspace/logs/jupyter.log` | JupyterLab kernel and notebook execution logs |
| `workspace/logs/system_info.log` | Boot diagnostics: Python version, RAM, CPU, disk stats |
| `workspace/logs/tensorbox-diagnostics.zip` | One-click zipped bundle generated via `tensorbox bug-report` |

---

## 🤖 2. Automated Dependency Management Strategy

To ensure dependencies never break unexpectedly when updating `requirements.txt`:

### A. Pre-Release Smoke Tests & Conflict Check
Before updating packages, run the automated verification script:
```bash
python scripts/check_dependencies.py
```
This runs `pip check` to ensure there are no version incompatibilities or broken transitive dependencies across all 200+ ML packages.

### B. Automated Weekly CI/CD Audit (`.github/workflows/ci.yml`)
We configured an automated GitHub Actions pipeline that:
1. **Runs weekly on schedule (every Monday at 00:00 UTC)**: Tests dependency graph resolution against new upstream package releases.
2. **Runs on every Pull Request and Push**: Executes `pytest tests` and builds the Docker container.
3. **Automated Multi-Arch Releases**: When you tag a release (e.g. `git tag v1.0.0 && git push origin v1.0.0`), GitHub Actions automatically builds and pushes `amd64` and `arm64` images to Docker Hub.

---

## 🚀 3. Releasing New Versions

1. **Update `requirements.in` and recompile:**
   ```bash
   python scripts/resolve_dependencies.py
   ```
2. **Verify tests and dependencies:**
   ```bash
   pytest tests
   python scripts/check_dependencies.py
   ```
3. **Tag and Push to Trigger Automated Build & Release:**
   ```bash
   git tag v1.0.0
   git push origin v1.0.0
   ```
   GitHub Actions will automatically build the multi-platform image and deploy it to Docker Hub without manual intervention!
