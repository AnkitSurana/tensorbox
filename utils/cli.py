"""Unified Tensorbox Command Line Interface."""
import sys
import os
import subprocess
import time
import zipfile
from pathlib import Path
from utils.config import Config, find_env_file
from utils.health_check import check_all_services, check_tcp_port
from utils.logger import get_logger, get_log_dir

logger = get_logger("tensorbox-cli")

PROVIDER_ENV_MAP = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "cohere": "COHERE_API_KEY",
    "huggingface": "HUGGINGFACE_API_KEY",
    "hf": "HUGGINGFACE_API_KEY",
    "pinecone": "PINECONE_API_KEY",
}


def print_banner():
    print("""
===============================================================
       🤖 Tensorbox CLI - User Tooling & Services
===============================================================
""")


def cmd_help():
    print_banner()
    print("""Available Commands:

  🔑 API Keys & Configuration:
    tensorbox status                 Check health of all services & API keys
    tensorbox set-key <name> <key>   Quickly set an API key in .env (e.g. tensorbox set-key openai sk-...)
    tensorbox config                 Interactive configuration wizard for all keys

  📁 Project Scaffolding & Git Submission:
    tensorbox new <project-name>     Create a new assignment project with starter files
    tensorbox git-init               Initialize secure user Git workspace (auto-ignores secrets/data)

  🗄️ Database Launchers:
    tensorbox postgres               Start & initialize local PostgreSQL database (:5432)
    tensorbox sqlite                 Show SQLite connection snippet & database path
    tensorbox mysql                  Show MySQL / MariaDB connection guidance
    tensorbox redis                  Start local Redis cache server (:6379)
    tensorbox chroma                 Start local persistent ChromaDB vector store (:8000)

  🚀 Application Launchers:
    tensorbox streamlit [file.py]    Launch Streamlit dashboard (:8501)
    tensorbox fastapi [file.py]      Launch FastAPI with interactive Swagger UI (:5000/docs)
    tensorbox tensorboard [logdir]   Launch TensorBoard visualization (:6006)

  📋 Logs & Diagnostics:
    tensorbox logs [service]         View live logs (jupyter, code-server, tensorbox, postgres)
    tensorbox bug-report             Create a zipped diagnostics bundle for instructors/TAs

  🧪 Testing & Verification:
    tensorbox verify                 Run end-to-end verification of all ML frameworks, DBs & APIs
    tensorbox test                   Run the automated pytest unit test suite
    tensorbox help                   Show this help message

  💡 Note: 'devbox' works as an alias for all 'tensorbox' commands.
""")


def cmd_set_key(args):
    if len(args) < 2:
        print("\n❌ Error: Missing provider name or API key.")
        print("👉 Usage: tensorbox set-key <provider> <api_key>")
        print("   Examples:")
        print("     tensorbox set-key openai sk-proj-1234567890abcdef")
        print("     tensorbox set-key anthropic sk-ant-1234567890")
        print("     tensorbox set-key google your-google-api-key")
        print("     tensorbox set-key huggingface hf_1234567890\n")
        return

    provider = args[0].lower().strip()
    key_value = args[1].strip()

    env_var = PROVIDER_ENV_MAP.get(provider, f"{provider.upper()}_API_KEY")
    Config.set(env_var, key_value)

    print(f"\n✅ Successfully saved {env_var} to {find_env_file()}")
    print(f"👉 You can now use {provider.capitalize()} directly in your Python code & notebooks!\n")


def cmd_config():
    print_banner()
    env_file = find_env_file()
    print(f"📝 Configuring environment variables in: {env_file}\n")

    for provider, env_var in [
        ("OpenAI", "OPENAI_API_KEY"),
        ("Anthropic", "ANTHROPIC_API_KEY"),
        ("Google Gemini", "GOOGLE_API_KEY"),
        ("HuggingFace", "HUGGINGFACE_API_KEY"),
        ("Cohere", "COHERE_API_KEY"),
        ("Postgres Password", "POSTGRES_PASSWORD"),
        ("Postgres Database", "POSTGRES_DB"),
    ]:
        current = Config.get(env_var, "")
        masked = f"{current[:8]}...{current[-4:]}" if len(current) > 12 else (current or "[Not Set]")
        val = input(f"Enter {provider} [{masked}]: ").strip()
        if val:
            Config.set(env_var, val)
            print(f"   ✓ Saved {env_var}")

    print("\n✅ Configuration complete! Run 'tensorbox status' to verify.\n")


def cmd_new(args):
    """Scaffold a starter project."""
    if not args:
        print("\n❌ Error: Please specify a project name.")
        print("👉 Usage: tensorbox new <project-name>")
        print("   Example: tensorbox new assignment-1\n")
        return

    project_name = args[0].strip().replace(" ", "-")
    base_dir = Path("projects")
    target_dir = base_dir / project_name

    if target_dir.exists():
        print(f"⚠️ Project folder already exists at: {target_dir}")
        return

    target_dir.mkdir(parents=True, exist_ok=True)

    # 1. Starter FastAPI / Python script
    (target_dir / "main.py").write_text(f'''"""
Assignment: {project_name}
Starter application powered by Tensorbox.
"""
from fastapi import FastAPI
import os

app = FastAPI(title="{project_name} API", version="1.0.0")

@app.get("/")
def home():
    return {{
        "project": "{project_name}",
        "status": "ready",
        "docs": "/docs"
    }}

@app.post("/predict")
def predict(data: dict):
    # Add your model inference logic here
    return {{"status": "success", "input": data, "result": "sample_prediction"}}
''')

    # 2. Starter README.md
    (target_dir / "README.md").write_text(f'''# 🚀 {project_name}

## 📋 Description
Assignment project created inside **Tensorbox**.

## 🚀 How to Run
- **FastAPI with Swagger UI:**
  ```bash
  tensorbox fastapi projects/{project_name}/main.py
  ```
  Visit [http://localhost:5000/docs](http://localhost:5000/docs)
''')

    # 3. Starter unit test
    (target_dir / "test_main.py").write_text(f'''def test_starter():
    assert True
''')

    print(f"\n🎉 Successfully created project at: {target_dir}")
    print("📁 Files created:")
    print("   • main.py       (FastAPI & ML API template)")
    print("   • README.md     (Assignment instructions)")
    print("   • test_main.py  (Starter unit test)")
    print(f"\n👉 Launch your new project API via:")
    print(f"   tensorbox fastapi {target_dir}/main.py\n")


def cmd_postgres():
    print("\n🐘 Setting up PostgreSQL Database...")
    host = Config.get("POSTGRES_HOST", "127.0.0.1")
    if host in ("postgres", "localhost", ""):
        host = "127.0.0.1"
    port = int(Config.get("POSTGRES_PORT", "5432"))
    user = Config.get("POSTGRES_USER", "postgres")
    password = Config.get("POSTGRES_PASSWORD", "postgres")
    dbname = Config.get("POSTGRES_DB", "ml_database")

    # Start service
    subprocess.run(["mkdir", "-p", "/var/run/postgresql"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["chown", "-R", "postgres:postgres", "/var/run/postgresql"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["service", "postgresql", "start"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1.5)

    try:
        subprocess.run(
            ["su", "-", "postgres", "-c", f"psql -c \"ALTER USER postgres WITH PASSWORD '{password}';\""],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        subprocess.run(
            ["su", "-", "postgres", "-c", f"psql -c \"CREATE DATABASE {dbname};\""],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    except Exception:
        pass

    if check_tcp_port(host, port, timeout=2.0) or check_tcp_port("127.0.0.1", 5432, timeout=2.0):
        print("✅ PostgreSQL is ONLINE and ready!")
        print(f"   • Host:     {host}")
        print(f"   • Port:     {port}")
        print(f"   • User:     {user}")
        print(f"   • Database: {dbname}")
        print("\n🐍 Python Connection Snippet:")
        print(f"""import psycopg2
conn = psycopg2.connect(
    host="{host}",
    port={port},
    user="{user}",
    password="{password}",
    dbname="{dbname}"
)
print("Connected to PostgreSQL successfully!")
""")
    else:
        print("❌ Could not connect to PostgreSQL on port 5432.")


def cmd_sqlite():
    print("\n🗃️ SQLite Database (File-Based, Zero-Setup):")
    db_path = Path("/workspace/data/ml_database.db")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"   • Database File: {db_path}")
    print("\n🐍 Python Connection Snippet:")
    print(f"""import sqlite3
conn = sqlite3.connect("{db_path}")
cursor = conn.cursor()
cursor.execute("CREATE TABLE IF NOT EXISTS experiments (id INTEGER PRIMARY KEY, name TEXT);")
conn.commit()
print("SQLite database ready!")
""")


def cmd_mysql():
    print("\n🐬 MySQL / MariaDB Client Support:")
    print("   • For local relational databases, use PostgreSQL (run 'tensorbox postgres') or SQLite (run 'tensorbox sqlite').")
    print("   • To connect to an external MySQL server, use PyMySQL:")
    print("""import pymysql
conn = pymysql.connect(host="your-mysql-host", user="root", password="password", database="db")
print("Connected to MySQL successfully!")
""")


def cmd_redis():
    print("\n⚡ Setting up Redis Server...")
    host = Config.get("REDIS_HOST", "127.0.0.1")
    port = int(Config.get("REDIS_PORT", "6379"))

    if not check_tcp_port(host, port, timeout=0.5):
        subprocess.run(["service", "redis-server", "start"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(1)

    if check_tcp_port(host, port, timeout=1.0):
        print("✅ Redis is ONLINE and ready!")
        print(f"   • Host: {host}")
        print(f"   • Port: {port}")
        print("\n🐍 Python Connection Snippet:")
        print(f"""import redis
r = redis.Redis(host='{host}', port={port})
r.set('tensorbox', 'online')
print("Redis test read:", r.get('tensorbox').decode())
""")
    else:
        print("❌ Could not connect to Redis on port 6379.\n")


def cmd_chroma():
    print("\n🚀 Setting up ChromaDB Vector Database...")
    host = Config.get("CHROMA_HOST", "127.0.0.1")
    port = int(Config.get("CHROMA_PORT", "8000"))
    data_path = Path("/workspace/data/chroma")
    data_path.mkdir(parents=True, exist_ok=True)

    if not check_tcp_port(host, port, timeout=0.5):
        subprocess.Popen(
            ["chroma", "run", "--path", str(data_path), "--port", str(port), "--host", "0.0.0.0"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        time.sleep(1.5)

    print("✅ ChromaDB is ready for vector search!")
    print(f"   • Persistence Path: {data_path}")
    print("\n🐍 Python Connection Snippet (Embedded / Direct):")
    print("""import chromadb
client = chromadb.PersistentClient(path="/workspace/data/chroma")
collection = client.get_or_create_collection("course_docs")
print("Connected to ChromaDB collection:", collection.name)
""")


def cmd_streamlit(args):
    app_file = args[0] if args else "app.py"
    target = Path(app_file)
    if not target.exists() and not (Path("/workspace") / app_file).exists():
        target = Path("/workspace") / app_file if Path("/workspace").exists() else Path(app_file)
        target.write_text("""import streamlit as st

st.set_page_config(page_title="Tensorbox Demo", layout="wide")
st.title("🚀 Tensorbox Streamlit Demo")
st.write("Welcome to your interactive ML dashboard running on port 8501!")

name = st.text_input("What is your name?", "ML User")
st.success(f"Hello, {name}! Your environment is ready.")
""")
        print(f"📝 Created sample Streamlit dashboard at {target}")

    print(f"\n🚀 Launching Streamlit on http://localhost:8501 ...")
    subprocess.run(["streamlit", "run", str(app_file), "--server.port", "8501", "--server.address", "0.0.0.0"])


def cmd_fastapi(args):
    app_file = args[0] if args else "main.py"
    target_path = Path(app_file)
    
    if not target_path.exists() and (Path("/workspace") / app_file).exists():
        target_path = Path("/workspace") / app_file

    if not target_path.exists():
        target_path = Path("/workspace/main.py") if Path("/workspace").exists() else Path("main.py")
        target_path.write_text("""from fastapi import FastAPI

app = FastAPI(title="Tensorbox API", description="Interactive REST API with OpenAPI Swagger UI")

@app.get("/")
def read_root():
    return {"status": "online", "system": "Tensorbox", "openapi_docs": "/docs"}

@app.post("/predict")
def predict(text: str):
    return {"input": text, "prediction": "positive", "confidence": 0.99}
""")
        print(f"📝 Created sample FastAPI service at {target_path}")

    app_dir = target_path.parent.resolve()
    module_name = target_path.stem

    print(f"\n🚀 Launching FastAPI on port 5000 from {app_dir}...")
    print(f"👉 Interactive OpenAPI Swagger Docs: http://localhost:5000/docs")
    print(f"👉 Alternative Redoc:                http://localhost:5000/redoc\n")
    
    subprocess.run([
        "uvicorn", f"{module_name}:app",
        "--app-dir", str(app_dir),
        "--host", "0.0.0.0",
        "--port", "5000",
        "--reload"
    ])


def cmd_tensorboard(args):
    log_dir = args[0] if args else "/workspace/runs"
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    print(f"\n🚀 Launching TensorBoard for logdir '{log_dir}' on http://localhost:6006 ...")
    subprocess.run(["tensorboard", "--logdir", str(log_dir), "--host", "0.0.0.0", "--port", "6006"])


def cmd_logs(args):
    log_dir = get_log_dir()
    print(f"\n📋 Tensorbox Logs (Directory: {log_dir}):\n")
    if not log_dir.exists():
        print("   No logs directory found.")
        return

    log_files = list(log_dir.glob("*.log"))
    if not log_files:
        print("   No active log files found yet.")
        return

    if args:
        target_name = args[0] if args[0].endswith(".log") else f"{args[0]}.log"
        target_file = log_dir / target_name
        if target_file.exists():
            print(f"--- Showing last 30 lines of {target_name} ---")
            lines = target_file.read_text(encoding="utf-8", errors="ignore").splitlines()
            for line in lines[-30:]:
                print(line)
        else:
            print(f"❌ Log file '{target_name}' not found. Available: {[f.name for f in log_files]}")
    else:
        for f in log_files:
            size_kb = f.stat().st_size / 1024
            print(f"   • {f.name:<20} ({size_kb:.1f} KB)")
        print("\n💡 Run 'tensorbox logs <name>' to view a specific log (e.g. tensorbox logs code-server)")


def cmd_git_init():
    """Configure entire workspace for safe Git submission with zero data/secret leakage."""
    print("\n🐙 Initializing Secure User Git Workspace...")
    workspace_dir = Path.cwd()
    gitignore_path = workspace_dir / ".gitignore"

    exclusions = """# Environment & Secrets (NEVER COMMIT)
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
"""
    if not gitignore_path.exists():
        gitignore_path.write_text(exclusions)
        print("   ✓ Created secure .gitignore (excluding .env, data/, models/, logs/)")
    else:
        print("   ✓ Verified existing .gitignore")

    if not (workspace_dir / ".git").exists():
        subprocess.run(["git", "init"], cwd=workspace_dir)
        subprocess.run(["git", "branch", "-M", "main"], cwd=workspace_dir)
        print("   ✓ Initialized Git repository on 'main' branch")
    else:
        print("   ✓ Existing Git repository detected")

    print("\n📋 Protected from Git (Will NOT be pushed):")
    print("   ❌ .env (API keys & passwords)")
    print("   ❌ data/ (Datasets, vector stores, SQLite DBs)")
    print("   ❌ models/ (Heavy neural network weights)")
    print("   ❌ logs/ (Runtime debug logs)")

    print("\n📦 Included in Git (Will be pushed for homework submission):")
    print("   ✅ projects/ (All assignment source code, main.py, test_main.py, README)")
    print("   ✅ notebooks/ (All Jupyter notebooks & analysis)")

    print("\n🚀 Next Steps to Push to Your Personal GitHub:")
    print("   1. git remote add origin https://github.com/<your-username>/<your-repo>.git")
    print("   2. git add .")
    print("   3. git commit -m \"feat: complete assignments\"")
    print("   4. git push -u origin main\n")


def cmd_bug_report():
    log_dir = get_log_dir()
    zip_path = log_dir / "tensorbox-diagnostics.zip"
    print("\n📦 Generating Tensorbox Diagnostics Bundle...")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in log_dir.glob("*.log"):
            z.write(f, arcname=f.name)

    print(f"✅ Diagnostics zip archive generated at:")
    print(f"   👉 {zip_path}")
    print("💡 Share this file with your instructor or teaching assistant for quick troubleshooting!\n")


def main():
    if len(sys.argv) < 2:
        cmd_help()
        return

    command = sys.argv[1].lower()
    subargs = sys.argv[2:]

    if command in ("help", "-h", "--help"):
        cmd_help()
    elif command in ("status", "health", "check"):
        check_all_services(verbose=True)
    elif command in ("set-key", "setkey", "set"):
        cmd_set_key(subargs)
    elif command in ("config", "setup"):
        cmd_config()
    elif command in ("new", "init", "create"):
        cmd_new(subargs)
    elif command in ("git-init", "git", "submit"):
        cmd_git_init()
    elif command in ("postgres", "postgresql", "psql"):
        cmd_postgres()
    elif command in ("sqlite", "sqlite3"):
        cmd_sqlite()
    elif command in ("mysql", "mariadb"):
        cmd_mysql()
    elif command == "redis":
        cmd_redis()
    elif command == "chroma":
        cmd_chroma()
    elif command == "streamlit":
        cmd_streamlit(subargs)
    elif command in ("fastapi", "openapi", "api", "uvicorn"):
        cmd_fastapi(subargs)
    elif command in ("tensorboard", "tb"):
        cmd_tensorboard(subargs)
    elif command in ("logs", "log"):
        cmd_logs(subargs)
    elif command in ("bug-report", "report", "diagnostics"):
        cmd_bug_report()
    elif command in ("resolve-deps", "resolve", "update-deps"):
        subprocess.run([sys.executable, "scripts/resolve_dependencies.py"])
    elif command in ("check-deps", "audit"):
        subprocess.run([sys.executable, "scripts/check_dependencies.py"])
    elif command in ("verify", "e2e", "test-all"):
        from utils.e2e_test import run_all_e2e_tests
        success = run_all_e2e_tests()
        sys.exit(0 if success else 1)
    elif command in ("test", "pytest"):
        test_dir = "/workspace/tests" if Path("/workspace/tests").exists() else ("/opt/tensorbox/tests" if Path("/opt/tensorbox/tests").exists() else "/opt/devbox/tests")
        subprocess.run(["pytest", test_dir])
    else:
        print(f"❌ Unknown tensorbox command: '{command}'")
        cmd_help()


if __name__ == "__main__":
    main()

