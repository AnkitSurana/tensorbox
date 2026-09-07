#!/usr/bin/env python3
"""Cross-platform local developer setup script for Tensorbox."""

import os
import sys
import shutil
import subprocess
import platform
from pathlib import Path


class Setup:
    def __init__(self):
        self.os_type = platform.system()
        self.script_dir = Path(__file__).parent.resolve()
        self.compose_cmd = self.detect_compose_command()
        print(f"\n{'='*50}\nTensorbox Environment Setup\nOS: {self.os_type}\n{'='*50}\n")

    def detect_compose_command(self):
        """Detect Docker Compose V2 ('docker compose') or V1 ('docker-compose')."""
        try:
            res = subprocess.run(["docker", "compose", "version"], capture_output=True, text=True)
            if res.returncode == 0:
                return ["docker", "compose"]
        except Exception:
            pass

        try:
            res = subprocess.run(["docker-compose", "version"], capture_output=True, text=True)
            if res.returncode == 0:
                return ["docker-compose"]
        except Exception:
            pass
        return ["docker", "compose"]

    def create_directories(self):
        print("📁 Creating workspace directories...")
        dirs = [
            'notebooks', 'data', 'datasets', 'projects', 'models',
            'utils', 'tests', 'logs'
        ]
        for dir_name in dirs:
            target = self.script_dir / dir_name
            target.mkdir(exist_ok=True)
            print(f"   ✓ {dir_name}")

        # Ensure .gitkeep in core content folders
        for keep_dir in ['notebooks', 'data', 'datasets', 'projects', 'models']:
            keep_file = self.script_dir / keep_dir / ".gitkeep"
            if not keep_file.exists():
                keep_file.touch()
        print("✅ Directories verified\n")

    def check_docker(self):
        print("🐳 Checking Docker...")
        try:
            result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"   ✓ {result.stdout.strip()}")
                cmd_str = " ".join(self.compose_cmd)
                print(f"   ✓ Using compose command: {cmd_str}\n")
                return True
        except FileNotFoundError:
            pass
        print("❌ Docker not found or not running!")
        print("   Please install and launch Docker Desktop: https://www.docker.com/products/docker-desktop\n")
        return False

    def setup_env(self):
        print("⚙️  Setting up .env configuration...")
        env_path = self.script_dir / ".env"
        if not env_path.exists():
            shutil.copy(self.script_dir / ".env.example", env_path)
            print("   ✓ Created .env from template")
        else:
            print("   ✓ Existing .env found")

    def build(self):
        print("\n📦 Building Tensorbox Docker image...\n")
        cmd = self.compose_cmd + ["build"]
        result = subprocess.run(cmd, cwd=self.script_dir)
        if result.returncode == 0:
            print("✅ Build complete\n")
            return True
        print("❌ Build failed. Please check the error messages above.\n")
        return False

    def start(self):
        print("🎬 Starting Tensorbox services...")
        cmd = self.compose_cmd + ["up", "-d"]
        result = subprocess.run(cmd, cwd=self.script_dir)
        if result.returncode == 0:
            print("✅ Containers started successfully\n")
            return True
        print("❌ Failed to start containers.\n")
        return False

    def print_info(self):
        print("=" * 60)
        print("🎉 TENSORBOX READY!")
        print("=" * 60)
        print("\n📖 Access Services:")
        print("   • VS Code (Browser):    http://localhost:8080")
        print("   • JupyterLab:           http://localhost:8888")
        print("   • ChromaDB:             http://localhost:8000")
        print("   • Streamlit Apps:       http://localhost:8501")
        print("   • FastAPI & OpenAPI:    http://localhost:5000/docs")
        print("\n💡 Logs directory: ./logs (mirrored on host machine)")
        print("=" * 60 + "\n")

    def run(self):
        try:
            self.create_directories()
            if not self.check_docker():
                sys.exit(1)
            self.setup_env()
            if not self.build():
                sys.exit(1)
            if not self.start():
                sys.exit(1)
            self.print_info()
        except KeyboardInterrupt:
            print("\n\n❌ Setup cancelled.\n")
            sys.exit(1)


if __name__ == "__main__":
    Setup().run()

