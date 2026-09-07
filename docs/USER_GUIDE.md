# 📖 Tensorbox User Guide

Welcome to **Tensorbox**! This is your pre-configured Machine Learning & Generative AI environment in a single Docker image. You **do not** need to install Python, configure complex dependencies, or pass any sensitive API keys in the Docker run command. Everything is managed easily inside your development workspace.

---

## ⚡ 1. Starting Your Tensorbox

### Step 1: Install Docker Desktop
Make sure [Docker Desktop](https://www.docker.com/products/docker-desktop) is installed and running on your machine (Windows with WSL2, macOS, or Linux).

### Step 2: Run the Standard Launch Command
Run this single, consistent command in your terminal (macOS/Linux) or PowerShell (Windows):

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

> [!NOTE]
> **Why `-p` instead of `--network host`?**  
> On macOS and Windows, Docker Desktop runs inside a virtual machine. `--network host` only attaches to the internal VM and will not expose ports to your Mac or Windows browser. Using `-p` guarantees that `http://localhost:8080` works seamlessly across Windows, macOS, and Linux.

### 💡 Do I Need an NVIDIA GPU?

**No. You do NOT need an NVIDIA GPU for the vast majority of coursework.**

* **95% of typical assignments run 100% on CPU:**
  * Classical Machine Learning (Scikit-learn, XGBoost, LightGBM, Pandas).
  * Vector Database similarity searches (ChromaDB, FAISS, Qdrant).
  * Neural text embeddings using models like `all-MiniLM-L6-v2` (embeds sentences in milliseconds on CPU).
  * Generative AI & RAG pipelines using OpenAI, Anthropic, Gemini, or Groq (models execute in the cloud).
  * Microservices & Dashboards (FastAPI OpenAPI docs, Streamlit, PostgreSQL, Redis).
* **When is a GPU actually needed?**
  * Training deep neural networks (e.g., ResNet, Vision Transformers) from scratch on large image/video datasets.
  * Fine-tuning large open-source language models locally (e.g., Llama 3 with LoRA/QLoRA).
* **If you have an NVIDIA GPU (Linux / Windows WSL2):**
  * Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) and add `--gpus all` to your `docker run` command for automatic CUDA acceleration.
* **If you have an Apple Silicon Mac (M1/M2/M3/M4) or Intel/AMD CPU:**
  * CPU performance and vectorized instructions handle all assignments smoothly. For heavy multi-hour model training assignments, instructors will typically provide access to Google Colab or university compute clusters.

---

## 💻 2. Accessing Your Environment

Open your web browser to access your tools:

| Interface | URL | Description |
| :--- | :--- | :--- |
| **💻 VS Code IDE** | **[http://localhost:8080](http://localhost:8080)** | In-browser VS Code with terminal, file manager, and Git tools |
| **📓 JupyterLab** | **[http://localhost:8888](http://localhost:8888)** | Interactive computational notebooks |

All files you create inside `/workspace` are automatically saved to the `workspace/` folder on your local computer.

---

## 🔑 3. Setting Up API Keys (OpenAI, Gemini, Anthropic, HuggingFace, Pinecone)

You can set up your keys inside the VS Code terminal using simple `devbox` commands or by editing `.env`:

### Method A: Quick CLI Command (Recommended)
Open the integrated terminal in VS Code (`Ctrl+\`` or `Cmd+\``) and run:

```bash
# Set your OpenAI Key:
devbox set-key openai sk-proj-your-key-here

# Set Anthropic Key:
devbox set-key anthropic sk-ant-your-key-here

# Set Google Gemini Key:
devbox set-key google your-google-api-key

# Set HuggingFace Token:
devbox set-key huggingface hf_your-token-here

# Set Pinecone Key (Optional for cloud vector DBs):
devbox set-key pinecone your-pinecone-api-key
```

### Method B: Interactive Configuration Wizard
Run:
```bash
devbox config
```
This wizard will prompt you for each API key and save them to `/workspace/.env`.

### Method C: Edit `.env` Directly in VS Code
Click on the `.env` file in the VS Code file explorer, paste your keys, and save (`Ctrl+S` / `Cmd+S`).

### Method D: Verifying Your Keys & Service Status
Run:
```bash
devbox status
```
If any key is missing, `devbox status` will provide exact instructions on how to set it.

---

## 🧠 4. Vector Databases & RAG Guide (Zero-Setup & Pre-Installed)

### Do I need to install Vector Databases?
**No.** All major vector database libraries, embedding engines, and RAG orchestration frameworks are **pre-installed** in your devbox container:
* `chromadb` (Embedded persistent vector database and client-server)
* `faiss-cpu` (Facebook AI Similarity Search for dense vectors)
* `qdrant-client` (Embedded and client-server vector search engine)
* `sentence-transformers` (Pre-trained neural embedding models)
* `langchain`, `langchain-community`, `langchain-openai` (RAG orchestration)
* `llama-index` (Data framework for LLM applications)
* `weaviate-client`, `pymilvus`, `pinecone-client` (Cloud vector store SDKs)

If your course requires an additional vector database (such as `lancedb`), you can install it instantly inside the VS Code terminal:
```bash
pip install lancedb
```

---

### A. ChromaDB (Local Persistent Vector Store)
ChromaDB is ideal for local prototyping. It runs entirely inside your container and saves vector collections directly to disk in your workspace.

#### 1. Embedded Persistent Mode (Recommended)
Vectors and metadata are saved in `/workspace/data/chroma/`:

```python
import chromadb

# 1. Initialize persistent storage in your workspace
client = chromadb.PersistentClient(path="/workspace/data/chroma")

# 2. Get or create a collection (uses default all-MiniLM-L6-v2 embeddings)
collection = client.get_or_create_collection(
    name="lecture_notes",
    metadata={"hnsw:space": "cosine"} # cosine, l2, or ip
)

# 3. Add text documents with metadata and unique IDs
collection.add(
    documents=[
        "Supervised learning trains models on labeled input-output pairs.",
        "Unsupervised learning finds hidden patterns in unlabeled datasets.",
        "Reinforcement learning trains agents using rewards and penalties."
    ],
    metadatas=[
        {"topic": "supervised", "difficulty": "beginner"},
        {"topic": "unsupervised", "difficulty": "intermediate"},
        {"topic": "reinforcement", "difficulty": "advanced"}
    ],
    ids=["doc1", "doc2", "doc3"]
)

# 4. Query top matching documents with optional metadata filtering
results = collection.query(
    query_texts=["How do neural networks learn from rewards and feedback?"],
    n_results=1,
    where={"topic": "reinforcement"} # Optional metadata filter
)

print("Top Match:", results["documents"][0][0])
print("Metadata:", results["metadatas"][0][0])
print("Distance:", results["distances"][0][0])
```

#### 2. Standalone Client-Server Mode
If you prefer running ChromaDB as an independent microservice:
1. Start the ChromaDB server in your terminal:
   ```bash
   devbox chroma
   ```
2. Connect from any Python script or notebook:
   ```python
   import chromadb
   client = chromadb.HttpClient(host="127.0.0.1", port=8000)
   collection = client.get_or_create_collection(name="shared_collection")
   ```

---

### B. FAISS (High-Performance Vector Indexing)
FAISS (Facebook AI Similarity Search) is designed for high-efficiency similarity search over dense vector embeddings.

```python
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# 1. Load an embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Sample dataset
documents = [
    "Gradient descent minimizes the loss function during model training.",
    "Transformers rely on self-attention mechanisms to process sequence data.",
    "Convolutional neural networks excel at image classification and detection.",
    "Recurrent neural networks process sequential data with internal memory states."
]

# 3. Generate 384-dimensional vector embeddings
embeddings = model.encode(documents).astype('float32')

# 4. Create an exact L2 distance index (IndexFlatL2)
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
print(f"Total vectors indexed: {index.ntotal}")

# 5. Search for the nearest neighbor
query = "How does attention work in transformer architectures?"
query_vector = model.encode([query]).astype('float32')

k = 2  # Top 2 results
distances, indices = index.search(query_vector, k)

for rank, (dist, idx) in enumerate(zip(distances[0], indices[0]), 1):
    print(f"Rank {rank}: [L2 Distance: {dist:.4f}] -> {documents[idx]}")

# 6. Save index to disk & reload whenever needed
faiss.write_index(index, "/workspace/data/faiss_index.bin")
loaded_index = faiss.read_index("/workspace/data/faiss_index.bin")
```

---

### C. Qdrant (Embedded On-Disk Vector Engine)
Qdrant offers vector indexing combined with payload filtering. You can run Qdrant in embedded mode with zero external containers:

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer

# 1. Initialize embedded Qdrant saved in workspace
client = QdrantClient(path="/workspace/data/qdrant")

# 2. Create collection
collection_name = "course_catalog"
client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
)

# 3. Generate embeddings & insert points with rich payload
model = SentenceTransformer('all-MiniLM-L6-v2')
courses = [
    {"id": 1, "text": "Deep Learning with PyTorch and CUDA", "level": "advanced", "dept": "CS"},
    {"id": 2, "text": "Introductory Data Science in Python", "level": "intro", "dept": "STAT"},
    {"id": 3, "text": "Natural Language Processing and LLMs", "level": "advanced", "dept": "CS"}
]

points = []
for item in courses:
    vector = model.encode(item["text"]).tolist()
    points.append(
        PointStruct(
            id=item["id"],
            vector=vector,
            payload={"text": item["text"], "level": item["level"], "dept": item["dept"]}
        )
    )

client.upsert(collection_name=collection_name, points=points)

# 4. Search with payload filter (only advanced CS courses)
query_vec = model.encode("language models and attention").tolist()
results = client.search(
    collection_name=collection_name,
    query_vector=query_vec,
    query_filter=Filter(
        must=[FieldCondition(key="dept", match=MatchValue(value="CS"))]
    ),
    limit=1
)

print("Top Match:", results[0].payload["text"])
print("Score:", results[0].score)
```

---

### D. LanceDB (Serverless On-Disk Vector DB)
If you want to use LanceDB:
1. Install: `pip install lancedb`
2. Usage:
```python
import lancedb

# Connect to disk storage in workspace
db = lancedb.connect("/workspace/data/lancedb")

# Create a table with vectors
data = [
    {"vector": [1.3, 2.4, 0.5], "item": "document A", "price": 10.5},
    {"vector": [0.1, 4.2, 3.1], "item": "document B", "price": 20.0},
]
table = db.create_table("my_table", data=data, mode="overwrite")

# Search nearest vectors
results = table.search([1.0, 2.0, 0.0]).limit(1).to_list()
print("LanceDB Result:", results)
```

---

### E. Cloud Vector Databases (Pinecone, Weaviate, Milvus)
If your course project connects to a managed cloud vector database:

```python
import os

# Pinecone Cloud
from pinecone import Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Weaviate Cloud
import weaviate
from weaviate.classes.init import Auth
client = weaviate.connect_to_weaviate_cloud(
    cluster_url=os.getenv("WEAVIATE_URL", "https://your-cluster.weaviate.network"),
    auth_credentials=Auth.api_key(os.getenv("WEAVIATE_API_KEY"))
)

# Milvus / Zilliz Cloud
from pymilvus import connections
connections.connect(
    alias="default",
    uri=os.getenv("MILVUS_URI", "https://your-zilliz-cluster.zillizcloud.com"),
    token=os.getenv("MILVUS_TOKEN")
)
```

---

### F. Complete End-to-End RAG Pipeline (LangChain + ChromaDB + OpenAI)

Here is an end-to-end Retrieval-Augmented Generation pipeline:

```python
import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Raw source material
corpus = """
Tensorbox is an all-in-one Docker environment designed for user machine learning development.
Users access VS Code at port 8080 and JupyterLab at port 8888.
PostgreSQL is available on port 5432 and can be initialized with 'devbox postgres'.
Redis cache runs on port 6379 via 'devbox redis'.
Vector databases including ChromaDB, FAISS, and Qdrant are pre-installed and ready out of the box.
API keys are stored securely in .env and configured using 'devbox set-key <provider> <key>'.
Assignments are pushed to personal GitHub accounts directly from /workspace/projects/<name>.
"""

# 2. Split document into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=120, chunk_overlap=20)
docs = splitter.create_documents([corpus])

# 3. Create persistent vector store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="/workspace/data/chroma_rag"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# 4. Build RAG prompt and chain
prompt_template = """Answer the question truthfully based only on the provided context:
Context:
{context}

Question: {question}
Answer:"""

prompt = ChatPromptTemplate.from_template(prompt_template)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 5. Ask questions
query = "How do users start PostgreSQL and configure their API keys?"
response = rag_chain.invoke(query)
print("--- RAG Answer ---")
print(response)
```

---

### G. LlamaIndex RAG Pipeline Example
If you are using LlamaIndex for RAG:

```python
from llama_index.core import VectorStoreIndex, Document

# 1. Create sample documents
documents = [
    Document(text="AIML Devbox provides in-browser VS Code on port 8080 and JupyterLab on 8888."),
    Document(text="ChromaDB and FAISS are pre-installed for vector search and RAG indexing."),
    Document(text="Run 'devbox status' inside the VS Code terminal to inspect all services and keys.")
]

# 2. Build index
index = VectorStoreIndex.from_documents(documents)

# 3. Query engine
query_engine = index.as_query_engine()
response = query_engine.query("What ports are used by VS Code and JupyterLab?")
print("LlamaIndex Response:", response)
```

---

## 🗄️ 5. Relational & Cache Databases (`devbox <service>`)

### A. PostgreSQL Database (`devbox postgres`)
Inside VS Code terminal, run:
```bash
devbox postgres
```
This automatically initializes the database and creates the `ml_database` schema on `127.0.0.1:5432`.

**Python Connection Example:**
```python
import psycopg2

conn = psycopg2.connect(
    host="127.0.0.1",
    port=5432,
    user="postgres",
    password="postgres",
    dbname="ml_database"
)
cursor = conn.cursor()
cursor.execute("CREATE TABLE IF NOT EXISTS experiments (id SERIAL PRIMARY KEY, name TEXT, accuracy FLOAT);")
cursor.execute("INSERT INTO experiments (name, accuracy) VALUES ('ResNet50', 0.94);")
conn.commit()
print("Saved experiment record successfully!")
```

### B. Redis In-Memory Cache (`devbox redis`)
Run:
```bash
devbox redis
```
**Python Connection Example:**
```python
import redis

r = redis.Redis(host='127.0.0.1', port=6379)
r.set('user_status', 'active')
print("Redis Value:", r.get('user_status').decode('utf-8'))
```

---

## 📁 6. Project Scaffolding (`devbox new`)

To create a new assignment folder with starter files:
```bash
devbox new assignment-1
```
This generates:
* `projects/assignment-1/main.py`: Starter FastAPI ML application.
* `projects/assignment-1/README.md`: Assignment instructions.
* `projects/assignment-1/test_main.py`: Starter unit test.

---

## 🚀 7. Launching Applications & Web Services

### A. FastAPI with Interactive OpenAPI Swagger Docs (Port 5000)
```bash
devbox fastapi
# Or with your specific project: devbox fastapi projects/assignment-1/main.py
```
👉 Interactive OpenAPI Swagger UI: **[http://localhost:5000/docs](http://localhost:5000/docs)**  
👉 Alternative Redoc: **[http://localhost:5000/redoc](http://localhost:5000/redoc)**

### B. Streamlit Dashboard Apps (Port 8501)
```bash
devbox streamlit
# Or with your specific project: devbox streamlit projects/assignment-1/app.py
```
👉 Live Dashboard: **[http://localhost:8501](http://localhost:8501)**

### C. TensorBoard Visualizations (Port 6006)
```bash
devbox tensorboard
```
👉 TensorBoard UI at **[http://localhost:6006](http://localhost:6006)**

---

## 📋 8. Logs & Getting Help (`devbox logs` / `devbox bug-report`)

All logs are automatically saved to the `workspace/logs/` folder on your host machine.

* **View Active Logs:**
  ```bash
  devbox logs
  # View specific log: devbox logs code-server
  ```
* **Generate a One-Click Diagnostics Report:**
  ```bash
  devbox bug-report
  ```
  This creates `workspace/logs/devbox-diagnostics.zip` on your computer, which you can easily send to your instructor or teaching assistant!

---

## 🐙 9. User Git Workflow (Submitting Projects & Code)

All your code and notebooks are stored permanently on your machine. You can submit either your **entire course workspace** or an **individual assignment project**.

### Method A: Submit the Entire Course Workspace (Recommended)
This submits all your `projects/` and `notebooks/` to a single personal repository while automatically excluding `.env` credentials, heavy datasets, and models.

1. Open the Terminal inside VS Code ([http://localhost:8080](http://localhost:8080)).
2. Run the automated Git initialization command:
   ```bash
   devbox git-init
   ```
   *This automatically creates a secure `.gitignore` that guarantees your private API keys (`.env`), datasets (`data/`), model weights (`models/`), and runtime logs (`logs/`) are never committed.*
3. Configure your personal Git details (once):
   ```bash
   git config --global user.name "Your Full Name"
   git config --global user.email "your.email@university.edu"
   ```
4. Connect your personal GitHub repository and push:
   ```bash
   git remote add origin https://github.com/<your-github-username>/<your-repo-name>.git
   git add .
   git commit -m "feat: initial workspace submission"
   git push -u origin main
   ```

---

### Method B: Submit an Individual Assignment
If your instructor requests a separate repository per assignment:
1. Navigate to that assignment's folder:
   ```bash
   cd /workspace/projects/assignment-1
   ```
2. Initialize and push:
   ```bash
   git init
   git remote add origin https://github.com/<your-github-username>/assignment-1.git
   git add .
   git commit -m "feat: complete assignment 1"
   git branch -M main
   git push -u origin main
   ```

---

## 🛑 10. Pausing & Resuming Your Tensorbox

* **To stop when you take a break:**
  ```bash
  docker stop tensorbox
  ```
* **To resume where you left off:**
  ```bash
  docker start tensorbox
  ```
All your workspace files, `.env` settings, and projects will be exactly as you left them!
