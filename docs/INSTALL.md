# 📥 Installation & Setup Guide

## 📋 System Requirements & Hardware Matrix

Before starting, verify your machine meets the supported chip and OS specifications:

| Platform | Supported Chips & Processors | Supported OS Versions | Minimum RAM | Recommended RAM |
| :--- | :--- | :--- | :--- | :--- |
| **🍎 Apple Mac** | • **Apple Silicon:** M1, M2, M3, M4 (Base, Pro, Max, Ultra)<br>• **Intel Mac:** Core i5 / i7 / i9 (2018 or newer) | macOS 12 Monterey<br>macOS 13 Ventura<br>macOS 14 Sonoma<br>macOS 15 Sequoia | 8 GB Unified Memory | 16 GB+ Unified Memory |
| **🪟 Windows** | • **Intel:** Core i5 / i7 / i9 (8th Gen or newer)<br>• **AMD:** Ryzen 5 / 7 / 9 (3000 series or newer)<br>• **ARM:** Snapdragon X Elite / Plus | Windows 10 (64-bit, 21H2+)<br>Windows 11 (Home / Pro) with WSL2 | 8 GB RAM | 16 GB+ RAM |
| **🐧 Linux** | • Any 64-bit x86_64 CPU (Intel / AMD, 4+ cores)<br>• Any ARM64 CPU (Ampere, Raspberry Pi 5 8GB, AWS Graviton) | Ubuntu 20.04, 22.04, 24.04 LTS<br>Debian 11 / 12<br>Fedora 38+ / Arch Linux | 8 GB RAM | 16 GB+ RAM |

* **Storage:** 15 GB free disk space (30 GB+ SSD recommended).
* **Virtualization:** Enabled in system BIOS/UEFI.
* **Software:** [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running.

---

## 👨‍💻 For Users

Users **do not** need to clone this repository. Simply install Docker Desktop and run the pre-built container.

### Step 1: Install & Configure Docker Desktop
* **Windows:** Install [Docker Desktop for Windows](https://docs.docker.com/desktop/install/windows-install/). Ensure **WSL 2 backend** is checked in Docker settings.
* **macOS:** Install [Docker Desktop for Mac](https://docs.docker.com/desktop/install/mac-install/). Under Settings > Resources, allocate at least 4-8 GB RAM.
* **Linux:** Install `docker.io` and `docker-compose-plugin`, then add your user to the docker group (`sudo usermod -aG docker $USER`).

### Step 2: Start Tensorbox
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

---

## 👨‍💻 For Developers & Instructors

To modify this repository, add dependencies, or publish releases:

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/tensorbox.git
   cd tensorbox
   ```
2. Run test suite:
   ```bash
   pytest tests
   ```
3. Build the Docker image:
   ```bash
   docker build -t tensorbox:latest .
   ```
4. Push to multi-arch registry:
   ```bash
   docker buildx build --platform linux/amd64,linux/arm64 -t <your-username>/tensorbox:latest -t <your-username>/tensorbox:1.0.0 --push .
   ```
