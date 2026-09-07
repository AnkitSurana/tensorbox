<div align="center">

# 🚀 Tensorbox
### An All-in-One, Zero-Setup Machine Learning & Generative AI Development Workstation

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg?logo=python&logoColor=white)](https://python.org)
[![Docker](https://img.shields.io/badge/Docker-Container-2496ED.svg?logo=docker&logoColor=white)](https://docker.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-FF6F00.svg?logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![LangChain](https://img.shields.io/badge/LangChain-Enabled-1C3C3C.svg)](https://langchain.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-OpenAPI-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

<p align="center">
  <b>Instant in-browser VS Code, JupyterLab, 200+ AI/ML libraries, embedded vector databases, and OpenAPI tooling.</b><br>
  <i>Users never have to install Python dependencies, configure CUDA, or manage port conflicts. Just run one Docker command and start coding!</i>
</p>

</div>

---

## 🌟 Instant Web Interfaces

Once launched, Tensorbox serves all tools through your local browser:

| Interface | URL | Description |
| :--- | :--- | :--- |
| **💻 VS Code IDE** | **[http://localhost:8080](http://localhost:8080)** | Full in-browser VS Code with terminal, extensions, and Git GUI |
| **📓 JupyterLab** | **[http://localhost:8888](http://localhost:8888)** | Computational notebooks for experiments and data science |
| **⚡ FastAPI / OpenAPI** | **[http://localhost:5000/docs](http://localhost:5000/docs)** | Interactive Swagger UI for REST ML prediction APIs |
| **📊 Streamlit** | **[http://localhost:8501](http://localhost:8501)** | Live interactive machine learning dashboards |
| **📈 TensorBoard** | **[http://localhost:6006](http://localhost:6006)** | Real-time neural network training visualizations |

## 📋 System Requirements & Hardware Matrix

Before running Tensorbox, verify that your computer meets the hardware and operating system specifications:

### 🖥️ Supported Processors & Operating Systems

| Platform | Supported Chips & Processors | Supported OS Versions | Minimum RAM | Recommended RAM |
| :--- | :--- | :--- | :--- | :--- |
| **🍎 Apple Mac** | • **Apple Silicon:** M1, M2, M3, M4 (Base, Pro, Max, Ultra)<br>• **Intel Mac:** Core i5 / i7 / i9 (2018 or newer) | macOS 12 Monterey<br>macOS 13 Ventura<br>macOS 14 Sonoma<br>macOS 15 Sequoia | 8 GB Unified Memory | 16 GB+ Unified Memory |
| **🪟 Windows** | • **Intel:** Core i5 / i7 / i9 (8th Gen or newer)<br>• **AMD:** Ryzen 5 / 7 / 9 (3000 series or newer)<br>• **ARM:** Snapdragon X Elite / Plus | Windows 10 (64-bit, 21H2+)<br>Windows 11 (Home / Pro) with WSL2 | 8 GB RAM | 16 GB+ RAM |
| **🐧 Linux** | • Any 64-bit x86_64 CPU (Intel / AMD, 4+ cores)<br>• Any ARM64 CPU (Ampere, Raspberry Pi 5 8GB, AWS Graviton) | Ubuntu 20.04, 22.04, 24.04 LTS<br>Debian 11 / 12<br>Fedora 38+ / Arch Linux | 8 GB RAM | 16 GB+ RAM |

### 💾 Storage & Virtualization Requirements
* **Disk Storage:** 15 GB minimum free space (30 GB+ SSD recommended for saving models, datasets, and persistent vector databases).
* **Virtualization:** Must be enabled in BIOS/UEFI (required by Docker Desktop and Windows WSL2).
* **Software:** [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows/Mac) or Docker Engine (Linux).
* **Browser:** Chrome, Firefox, Safari, Edge, or Brave.

> [!NOTE]
> **Windows Users:** Ensure **Virtualization** is enabled in your BIOS/UEFI and WSL2 is selected in Docker Desktop Settings (`Settings > General > Use the WSL 2 based engine`).

### 💡 Do I Need an NVIDIA GPU?

**No, an NVIDIA GPU is NOT required for most coursework:**

* **95% of AI/ML Tasks Run 100% on CPU:**
  * Classical machine learning (Scikit-learn, XGBoost, LightGBM, Pandas).
  * Vector databases & semantic search (ChromaDB, FAISS, Qdrant).
  * Neural text embeddings using models like `all-MiniLM-L6-v2` (runs in milliseconds on CPU).
  * Generative AI & RAG pipelines connecting to OpenAI, Anthropic, Gemini, or Groq (inference runs in the cloud).
  * Web APIs & UI dashboards (FastAPI, Streamlit, PostgreSQL, Redis).
* **When is an NVIDIA GPU helpful?**
  * Training deep neural networks (PyTorch/TensorFlow) from scratch on large image/video datasets.
  * Fine-tuning large open-source language models locally (e.g., Llama 3 8B with LoRA).
* **If you have an NVIDIA GPU:**
  * On Linux / Windows WSL2 with the NVIDIA Container Toolkit installed, simply append `--gpus all` to your `docker run` command for automatic CUDA acceleration.
* **If you do not have an NVIDIA GPU:**
  * Develop and test your code locally in Tensorbox, then run heavy training scripts on free cloud GPU environments (Google Colab, Kaggle, or your university compute cluster).

---

## 👨‍💻 For Users: 1-Command Launch (Zero Clone)

You do **not** need to clone this repository. Simply ensure [Docker Desktop](https://www.docker.com/products/docker-desktop) is running on your computer, then execute:

### 🍎 macOS / 🐧 Linux:
```bash
docker run -d \
  --name tensorbox \
  -p 8080:8080 \
  -p 8888:8888 \
  -p 8501:8501 \
  -p 5000:5000 \
  -p 6006:6006 \
  -v "${PWD}/workspace:/workspace" \
  tensorbox:latest
```

### 🪟 Windows (PowerShell):
```powershell
docker run -d `
  --name tensorbox `
  -p 8080:8080 `
  -p 8888:8888 `
  -p 8501:8501 `
  -p 5000:5000 `
  -p 6006:6006 `
  -v "${PWD}/workspace:/workspace" `
  tensorbox:latest
```

> [!TIP]
> Open **[http://localhost:8080](http://localhost:8080)** in your browser to immediately begin coding in VS Code!

---

## 🔑 In-Container Key Setup & CLI Tooling

No private keys are passed in the `docker run` command. Configure API keys safely inside the VS Code terminal using the pre-installed `tensorbox` CLI (also aliased as `devbox`):

```bash
# Set your API keys directly into .env
tensorbox set-key openai sk-proj-your-key-here
tensorbox set-key anthropic sk-ant-your-key-here
tensorbox set-key google your-gemini-key

# Or launch the interactive configuration wizard:
tensorbox config

# Check live health of all services & API keys:
tensorbox status
```

---

## 🗄️ Self-Contained Databases & Services

All database engines are embedded directly in the Tensorbox container. Run single commands from the VS Code terminal to start and test them:

```bash
# 🐘 PostgreSQL: Automatically creates 'ml_database' on 127.0.0.1:5432
tensorbox postgres

# ⚡ Redis: Launches in-memory cache server on 127.0.0.1:6379
tensorbox redis

# 🚀 ChromaDB: Launches persistent vector store on 127.0.0.1:8000
tensorbox chroma

# 📁 Scaffold a new assignment project with starter files:
tensorbox new assignment-1

# ⚡ Launch FastAPI with interactive Swagger UI (:5000/docs):
tensorbox fastapi projects/assignment-1/main.py

# 📊 Launch Streamlit dashboard (:8501):
tensorbox streamlit projects/assignment-1/app.py
```

---

## 🐙 User Git Submission Workflow

Users can submit their **entire workspace** or an **individual assignment/project folder** to their personal GitHub account.

### 🌟 Submit Entire Workspace (Recommended)
Run `tensorbox git-init` to set up an automated, secure `.gitignore` that guarantees `.env` API keys, large datasets (`data/`), models (`models/`), and logs (`logs/`) are never pushed:

```bash
# 1. Initialize secure workspace (run inside VS Code terminal):
tensorbox git-init

# 2. Add your personal GitHub repo and push:
git remote add origin https://github.com/<your-username>/<your-repo>.git
git add .
git commit -m "feat: submit coursework"
git push -u origin main
```

---

## 📋 Host-Visible Logs & Diagnostics

If you run into issues, all container logs are mirrored directly on your physical computer in `workspace/logs/`:
* `workspace/logs/tensorbox.log`: Unified tool and script execution logs.
* `workspace/logs/code-server.log`: VS Code server and extension logs.
* `workspace/logs/jupyter.log`: JupyterLab notebook runtime logs.
* `workspace/logs/system_info.log`: Container RAM, CPU architecture, and Python stats.

> [!NOTE]
> **Need help from a TA?** Run `tensorbox bug-report` inside the terminal to create a one-click `workspace/logs/tensorbox-diagnostics.zip` bundle ready to share!

---

## 👨‍💻 For Instructors & Developers (Repository Maintainers)

This repository is maintained by instructors to build, test, and release Docker images.

### 🤖 Automated Dependency Solving
To add/update packages without version conflicts:
1. Edit [requirements.in](requirements.in) with flexible ranges.
2. Run the automated SAT resolver:
   ```bash
   python scripts/resolve_dependencies.py
   ```
3. Run the automated conflict and smoke test:
   ```bash
   python scripts/check_dependencies.py
   ```

### 🚢 Building & Releasing
```bash
# Run test suite
pytest tests

# Build image locally
docker build -t tensorbox:latest .

# Release via git tag (triggers automated multi-arch GitHub Actions build):
git tag v1.0.0 && git push origin v1.0.0
```

---

## 📚 Documentation Index

All in-depth documentation is organized in the [`docs/`](docs/) directory:

| Guide | Description |
| :--- | :--- |
| **[📖 User Guide](docs/USER_GUIDE.md)** | Step-by-step tutorial: starting, LLMs, OpenAPI, Streamlit, databases, and Git push. |
| **[🛠️ Developer Guide](docs/DEVELOPER_GUIDE.md)** | Architecture, dependency solver, CI/CD pipeline, and release process. |
| **[💻 Commands Reference](docs/COMMANDS.md)** | Comprehensive cheatsheet for host Docker and in-container `tensorbox` commands. |
| **[📥 Installation & Troubleshooting](docs/INSTALL.md)** | OS-specific Docker setup steps and common troubleshooting fixes. |
| **[🤝 Contributing Guide](docs/CONTRIBUTING.md)** | Guidelines for PR testing, code style, and test validation. |
| **[🔒 Security Policy](docs/SECURITY.md)** | Credential isolation, `.env` safety, and vulnerability reporting. |

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
