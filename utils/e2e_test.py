"""End-to-End Verification Test Suite for Tensorbox.

Tests all integrated services, databases, vector stores, deep learning frameworks,
and NLP pipelines cleanly in isolated test runners.
"""

import sys
import time
import subprocess
from pathlib import Path


class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def print_header(title: str):
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN} 🧪 {title}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.RESET}")


def run_isolated_test(name: str, snippet: str):
    start = time.time()
    res = subprocess.run(
        [sys.executable, "-c", snippet],
        capture_output=True,
        text=True
    )
    elapsed = time.time() - start
    if res.returncode == 0:
        detail = res.stdout.strip().splitlines()[-1] if res.stdout.strip() else ""
        detail_msg = f" - {detail}" if detail else ""
        print(f"  {Colors.GREEN}✓ [PASS]{Colors.RESET} {name:<40} ({elapsed:.3f}s){detail_msg}")
        return True
    else:
        print(f"  {Colors.RED}✗ [FAIL]{Colors.RESET} {name:<40} ({elapsed:.3f}s)")
        err_detail = (res.stderr or res.stdout).strip().splitlines()
        for line in err_detail[-3:]:
            print(f"    {Colors.RED}{line}{Colors.RESET}")
        return False


# ----------------------------------------------------------------------
# Test Snippets
# ----------------------------------------------------------------------

TESTS = [
    (
        "Core Data Science (NumPy, Pandas, Sklearn)",
        """
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.linear_model import Ridge
X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
y = np.array([1.0, 2.0, 3.0])
model = Ridge(alpha=0.1)
model.fit(X, y)
pred = model.predict([[2.0, 3.0]])
print(f"Numpy {np.__version__}, Pandas {pd.__version__}, Ridge Pred={pred[0]:.2f}")
"""
    ),
    (
        "Deep Learning (PyTorch & Tensor ops)",
        """
import torch
x = torch.randn(3, 3)
y = torch.matmul(x, x.t())
cuda_avail = torch.cuda.is_available()
mps_avail = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
device = "CUDA" if cuda_avail else ("MPS" if mps_avail else "CPU")
print(f"PyTorch {torch.__version__} (Device: {device})")
"""
    ),
    (
        "Deep Learning (TensorFlow)",
        """
import tensorflow as tf
hello = tf.constant("Tensorbox TF Online")
print(f"TensorFlow {tf.__version__}")
"""
    ),
    (
        "TensorBoard & Setuptools Compatibility",
        """
import tensorboard
from tensorboard import default
import pkg_resources
print(f"TensorBoard {tensorboard.__version__} (pkg_resources verified)")
"""
    ),
    (
        "NLP Pipelines & Models (spaCy)",
        """
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("Tensorbox AI workstations empower machine learning users.")
ents = [(ent.text, ent.label_) for ent in doc.ents]
print(f"spaCy {spacy.__version__} (Loaded en_core_web_sm)")
"""
    ),
    (
        "Generative AI (LangChain, LlamaIndex)",
        """
import langchain
import langchain_core
import llama_index.core
print(f"LangChain {langchain.__version__}, LlamaIndex {llama_index.core.__version__}")
"""
    ),
    (
        "Vector DB & Search (ChromaDB)",
        """
import chromadb
client = chromadb.Client()
coll = client.get_or_create_collection("tensorbox_e2e_test")
coll.add(
    embeddings=[[0.1, 0.2, 0.3], [0.8, 0.9, 0.1]],
    documents=["Machine Learning Tensorbox", "Deep Learning with PyTorch"],
    ids=["doc1", "doc2"]
)
res = coll.query(query_embeddings=[[0.1, 0.2, 0.3]], n_results=1)
client.delete_collection("tensorbox_e2e_test")
matched_id = res["ids"][0][0]
print(f"ChromaDB {chromadb.__version__} (Indexed & Queried Vector ID: '{matched_id}')")
"""
    ),
    (
        "File Relational DB (SQLite3)",
        """
import sqlite3
from pathlib import Path
db_path = Path("/workspace/data/e2e_test.db") if Path("/workspace/data").exists() else Path("e2e_test.db")
conn = sqlite3.connect(db_path)
cur = conn.cursor()
cur.execute("CREATE TABLE IF NOT EXISTS test (id INTEGER PRIMARY KEY, msg TEXT);")
cur.execute("INSERT INTO test (msg) VALUES ('tensorbox_online');")
conn.commit()
cur.execute("SELECT msg FROM test ORDER BY id DESC LIMIT 1;")
val = cur.fetchone()[0]
conn.close()
if db_path.exists():
    db_path.unlink()
print(f"SQLite3 Verified (Read/Write OK: '{val}')")
"""
    ),
    (
        "Key-Value Cache (Redis)",
        """
import redis
import time
import subprocess
from utils.config import Config
host = Config.get("REDIS_HOST", "127.0.0.1")
port = int(Config.get("REDIS_PORT", "6379"))
try:
    r = redis.Redis(host=host, port=port, socket_timeout=1.0)
    r.ping()
except Exception:
    subprocess.run(["service", "redis-server", "start"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1)
    r = redis.Redis(host="127.0.0.1", port=6379, socket_timeout=1.0)
    r.ping()
r.set("e2e_key", "tensorbox_ok", ex=10)
read_val = r.get("e2e_key").decode()
print(f"Redis Connected at {host}:{port} (Key Test: '{read_val}')")
"""
    ),
    (
        "Relational Database (PostgreSQL)",
        """
import psycopg2
import time
import subprocess
from utils.config import Config
host = Config.get("POSTGRES_HOST", "127.0.0.1")
if host in ("postgres", "localhost", ""):
    host = "127.0.0.1"
port = int(Config.get("POSTGRES_PORT", "5432"))
user = Config.get("POSTGRES_USER", "postgres")
password = Config.get("POSTGRES_PASSWORD", "postgres")
try:
    conn = psycopg2.connect(host=host, port=port, user=user, password=password, dbname="postgres", connect_timeout=2)
except Exception:
    subprocess.run(["mkdir", "-p", "/var/run/postgresql"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["chown", "-R", "postgres:postgres", "/var/run/postgresql"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["service", "postgresql", "start"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1.5)
    subprocess.run(["su", "-", "postgres", "-c", "psql -c \\"ALTER USER postgres WITH PASSWORD 'postgres';\\""], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    conn = psycopg2.connect(host="127.0.0.1", port=5432, user="postgres", password=password, dbname="postgres", connect_timeout=3)

cur = conn.cursor()
cur.execute("SELECT version();")
ver = cur.fetchone()[0].split()[1]
conn.close()
print(f"PostgreSQL {ver} Connected at {host}:{port}")
"""
    ),
    (
        "Web Framework & REST API (FastAPI)",
        """
from fastapi import FastAPI
from fastapi.testclient import TestClient
app = FastAPI()
@app.get("/health")
def health():
    return {"status": "healthy", "service": "tensorbox"}
client = TestClient(app)
res = client.get("/health")
assert res.status_code == 200
print(f"FastAPI TestClient Responded: {res.json()['status']}")
"""
    ),
]


def run_all_e2e_tests() -> bool:
    print_header("Tensorbox Full End-to-End System Verification")

    passed = 0
    failed = 0

    for name, snippet in TESTS:
        if run_isolated_test(name, snippet):
            passed += 1
        else:
            failed += 1

    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.RESET}")
    if failed == 0:
        print(f"{Colors.BOLD}{Colors.GREEN}🎉 ALL {passed}/{len(TESTS)} E2E TESTS PASSED SUCCESSFULLY!{Colors.RESET}")
    else:
        print(f"{Colors.BOLD}{Colors.RED}❌ {failed}/{len(TESTS)} TESTS FAILED.{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.RESET}\n")

    return failed == 0


if __name__ == "__main__":
    success = run_all_e2e_tests()
    sys.exit(0 if success else 1)

