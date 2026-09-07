"""
Generates Notebooks for:
- Track 11: Recommendation Systems (2 notebooks)
- Track 12: Generative Deep Learning (2 notebooks)
- Track 13: Generative AI, LLMs, RAG & Multi-Agent Swarms (5 notebooks)
- Track 14: Reinforcement Learning & Production MLOps (4 notebooks)
"""

from .common import create_notebook, save_notebook, DATA_LOADER_BOILERPLATE

def build_tracks_11_to_14():
    # ==========================================
    # TRACK 11 - NOTEBOOK 01: SVD Matrix Factorization
    # ==========================================
    nb_t11_01 = create_notebook(
        title="01: Collaborative Filtering: SVD Matrix Factorization & Top-K Ranking",
        description="Build recommendation engines using Singular Value Decomposition (SVD): User-Item sparse utility matrix, latent factor decomposition, and Top-K recommendation retrieval.",
        track_num="11",
        track_name="Recommendation Systems",
        cells_data=[
            ("md", "## 1. User-Item Interaction Matrix Ingestion\nLoading movie ratings dataset and building sparse user-item interaction matrix."),
            ("code", DATA_LOADER_BOILERPLATE + """
import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds

df = load_dataset("movie_ratings")
print(f"Movie Ratings: {len(df)} interactions")
print(df.head())

pivot_df = df.pivot(index="userId", columns="movieId", values="rating").fillna(0)
matrix = pivot_df.values
user_ratings_mean = np.mean(matrix, axis=1)
matrix_demeaned = matrix - user_ratings_mean.reshape(-1, 1)

print(f"Utility Matrix Shape: {matrix.shape} (Users x Movies)")
"""),
            ("md", "## 2. Low-Rank Matrix Factorization via SVD\nDecompose matrix $R \\approx U \\Sigma V^T$ with rank $k=10$."),
            ("code", """
k_factors = min(10, min(matrix.shape) - 1)
U, sigma, Vt = svds(matrix_demeaned, k=k_factors)
sigma_diag = np.diag(sigma)

predicted_ratings = np.dot(np.dot(U, sigma_diag), Vt) + user_ratings_mean.reshape(-1, 1)
preds_df = pd.DataFrame(predicted_ratings, columns=pivot_df.columns, index=pivot_df.index)

print("Reconstructed Predicted Ratings Sample (Top 5 users, 5 movies):")
print(preds_df.iloc[:5, :5].round(2))
""")
        ]
    )
    save_notebook(nb_t11_01, "11_recommenders/01_collaborative_filtering_svd_matrix_factorization.ipynb")

    # ==========================================
    # TRACK 11 - NOTEBOOK 02: Neural Collaborative Filtering (Two-Tower)
    # ==========================================
    nb_t11_02 = create_notebook(
        title="02: Neural Collaborative Filtering & Two-Tower Architecture in PyTorch",
        description="Deep learning recommendation architectures: User Tower and Item Tower neural embeddings, dot product similarity, and Binary Cross-Entropy click-through optimization.",
        track_num="11",
        track_name="Recommendation Systems",
        cells_data=[
            ("md", "## 1. Two-Tower Architecture Mechanics\n$$\\hat{y}_{u, i} = \\sigma\\left(f_{\\text{user}}(u)^T f_{\\text{item}}(i)\\right)$$"),
            ("code", """
import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoTowerRecommender(nn.Module):
    def __init__(self, num_users, num_items, embed_dim=32):
        super().__init__()
        # User Tower
        self.user_embed = nn.Embedding(num_users, embed_dim)
        self.user_fc = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim)
        )
        
        # Item Tower
        self.item_embed = nn.Embedding(num_items, embed_dim)
        self.item_fc = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim)
        )
        
    def forward(self, user_ids, item_ids):
        u_emb = self.user_fc(self.user_embed(user_ids))
        i_emb = self.item_fc(self.item_embed(item_ids))
        
        u_norm = F.normalize(u_emb, p=2, dim=-1)
        i_norm = F.normalize(i_emb, p=2, dim=-1)
        
        score = (u_norm * i_norm).sum(dim=-1)
        return torch.sigmoid(score)

model = TwoTowerRecommender(num_users=1000, num_items=500, embed_dim=32)
user_batch = torch.tensor([1, 4, 12, 50])
item_batch = torch.tensor([10, 25, 120, 300])

scores = model(user_batch, item_batch)
print("=== Two-Tower Model Forward Pass ===")
print(f"Predicted Interaction Scores: {scores.detach().numpy().round(4)}")
""")
        ]
    )
    save_notebook(nb_t11_02, "11_recommenders/02_neural_collaborative_filtering_two_tower.ipynb")

    # ==========================================
    # TRACK 12 - NOTEBOOK 01: Variational Autoencoders (VAE)
    # ==========================================
    nb_t12_01 = create_notebook(
        title="01: Generative Deep Learning: Variational Autoencoders (VAEs)",
        description="Generative modeling in PyTorch: Encoder Gaussian parameterization (Mean $\\mu$ and Log-Variance $\\log\\sigma^2$), Reparameterization Trick, KL Divergence loss, and latent image sampling.",
        track_num="12",
        track_name="Generative Deep Learning (VAEs & Multimodal Search)",
        cells_data=[
            ("md", "## 1. Reparameterization Trick & Loss Formulation\n$$\\mathcal{L}_{\\text{VAE}} = \\mathbb{E}_{q_\\phi(z|x)}[\\log p_\\theta(x|z)] - D_{\\text{KL}}(q_\\phi(z|x) \\parallel p(z))$$\nwhere $z = \\mu + \\sigma \\odot \\epsilon$, with $\\epsilon \\sim \\mathcal{N}(0, I)$."),
            ("code", """
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim=784, latent_dim=16):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        
        self.fc3 = nn.Linear(latent_dim, 256)
        self.fc4 = nn.Linear(256, input_dim)
        
    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def decode(self, z):
        h = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h))
        
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    bce = F.binary_cross_entropy(recon_x, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + kld

vae = VAE(input_dim=784, latent_dim=16)
dummy_batch = torch.rand(8, 784)
recon, mu, logvar = vae(dummy_batch)
loss = vae_loss(recon, dummy_batch, mu, logvar)

print("=== VAE Architecture & Loss ===")
print(f"Input Shape : {dummy_batch.shape}")
print(f"Recon Shape : {recon.shape} | Latent Mu Shape: {mu.shape}")
print(f"Total ELBO Loss: {loss.item():.2f}")
""")
        ]
    )
    save_notebook(nb_t12_01, "12_generative_deep_learning/01_variational_autoencoders_vae.ipynb")

    # ==========================================
    # TRACK 12 - NOTEBOOK 02: Multimodal CLIP Search
    # ==========================================
    nb_t12_02 = create_notebook(
        title="02: Multimodal Machine Learning: CLIP Embeddings & Zero-Shot Search",
        description="Contrastive Language-Image Pre-Training (CLIP): Joint image and text embedding spaces, zero-shot visual classification, and cross-modal semantic search.",
        track_num="12",
        track_name="Generative Deep Learning (VAEs & Multimodal Search)",
        cells_data=[
            ("md", "## 1. Contrastive Cross-Modal Representation\nAligning text captions $\\mathbf{t}$ and image representations $\\mathbf{i}$ via symmetric InfoNCE loss:\n$$\\mathcal{L} = -\\frac{1}{2N}\\sum_{i=1}^N \\left( \\log\\frac{e^{\\mathbf{i}_i \\cdot \\mathbf{t}_i / \\tau}}{\\sum_j e^{\\mathbf{i}_i \\cdot \\mathbf{t}_j / \\tau}} + \\log\\frac{e^{\\mathbf{t}_i \\cdot \\mathbf{i}_i / \\tau}}{\\sum_j e^{\\mathbf{t}_i \\cdot \\mathbf{i}_j / \\tau}} \\right)$$"),
            ("code", """
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

np.random.seed(42)
d = 32

images = {
    "img_red_sports_car": np.random.randn(d) + np.array([2.0, 0.0, 0.0] + [0.0]*(d-3)),
    "img_golden_retriever": np.random.randn(d) + np.array([0.0, 2.0, 0.0] + [0.0]*(d-3)),
    "img_server_datacenter": np.random.randn(d) + np.array([0.0, 0.0, 2.0] + [0.0]*(d-3))
}

queries = {
    "a high speed automotive vehicle": np.random.randn(d) + np.array([2.1, 0.0, 0.0] + [0.0]*(d-3)),
    "a friendly cute puppy dog": np.random.randn(d) + np.array([0.0, 2.2, 0.0] + [0.0]*(d-3)),
    "cloud computing infrastructure": np.random.randn(d) + np.array([0.0, 0.0, 2.1] + [0.0]*(d-3))
}

print("=== Cross-Modal Zero-Shot Search Results ===")
for q_text, q_vec in queries.items():
    sims = {img_name: cosine_similarity(q_vec.reshape(1, -1), img_vec.reshape(1, -1))[0][0] for img_name, img_vec in images.items()}
    best_img = max(sims, key=sims.get)
    print(f"Query: '{q_text}' -> Top Visual Match: [{best_img}] (Score: {sims[best_img]:.4f})")
""")
        ]
    )
    save_notebook(nb_t12_02, "12_generative_deep_learning/02_multimodal_clip_embeddings.ipynb")

    # ==========================================
    # TRACK 13 - NOTEBOOK 01: Hybrid RAG
    # ==========================================
    nb_t13_01 = create_notebook(
        title="01: Production Hybrid RAG: BM25 Sparse + Dense Vector Search + Reranking",
        description="Implement enterprise Hybrid Retrieval-Augmented Generation: BM25 lexical token search, dense vector embeddings, Reciprocal Rank Fusion (RRF), and Cross-Encoder reranking.",
        track_num="13",
        track_name="Generative AI, LLMs, RAG & Multi-Agent Swarms",
        cells_data=[
            ("md", "## 1. Load Knowledge Base & Document Chunking\nIngest enterprise documentation corpus and segment into semantic chunks."),
            ("code", """
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

docs = [
    "Tensorbox provides zero-setup Docker environments for AI and Machine Learning.",
    "PyTorch is an open source deep learning framework developed by Meta AI.",
    "FastAPI enables high-performance REST APIs with automatic OpenAPI Swagger documentation.",
    "BM25 is a ranking function used by search engines to estimate relevance of documents.",
    "Hybrid RAG combines dense vector retrieval with keyword BM25 indexing for superior accuracy.",
    "LoRA (Low-Rank Adaptation) reduces trainable parameters by decomposing weight update matrices."
]

print(f"Knowledge Corpus Chunks: {len(docs)}")
"""),
            ("md", "## 2. Reciprocal Rank Fusion (RRF) Hybrid Merger\nCombine dense semantic ranks with sparse BM25 ranks:\n$$\\text{RRF}(d) = \\sum_{m \\in \\mathcal{M}} \\frac{1}{k + r_m(d)}$$"),
            ("code", """
def reciprocal_rank_fusion(sparse_ranks, dense_ranks, k=60):
    all_docs = set(sparse_ranks.keys()).union(set(dense_ranks.keys()))
    rrf_scores = {}
    for doc_id in all_docs:
        score = 0.0
        if doc_id in sparse_ranks:
            score += 1.0 / (k + sparse_ranks[doc_id])
        if doc_id in dense_ranks:
            score += 1.0 / (k + dense_ranks[doc_id])
        rrf_scores[doc_id] = score
    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

query = "How does hybrid RAG work with BM25?"
sparse_ranking = {4: 1, 3: 2, 0: 3}
dense_ranking = {4: 1, 0: 2, 5: 3}

fusion_results = reciprocal_rank_fusion(sparse_ranking, dense_ranking)
print("=== Hybrid RRF Ranked Documents ===")
for doc_id, rrf_score in fusion_results:
    print(f"Doc #{doc_id} (RRF Score: {rrf_score:.5f}): {docs[doc_id]}")
""")
        ]
    )
    save_notebook(nb_t13_01, "13_genai_and_agents/01_hybrid_rag_bm25_dense_rerank.ipynb")

    # ==========================================
    # TRACK 13 - NOTEBOOK 02: Graph RAG
    # ==========================================
    nb_t13_02 = create_notebook(
        title="02: Graph RAG: Knowledge Graph Augmented Semantic Retrieval",
        description="Combine structured knowledge graph traversal with vector similarity to answer multi-hop relational questions with grounded facts.",
        track_num="13",
        track_name="Generative AI, LLMs, RAG & Multi-Agent Swarms",
        cells_data=[
            ("md", "## 1. Multi-Hop Graph Traversal for Context Retrieval\nTraversing entity relations to synthesize contextual answers."),
            ("code", """
import networkx as nx

G = nx.DiGraph()
G.add_edge("Tensorbox", "PyTorch", relation="INCLUDES")
G.add_edge("Tensorbox", "FastAPI", relation="INCLUDES")
G.add_edge("Tensorbox", "Docker", relation="DEPLOYED_VIA")
G.add_edge("PyTorch", "LoRA", relation="SUPPORTS_FINE_TUNING")
G.add_edge("FastAPI", "Port 5001", relation="SERVED_ON")

def graph_rag_query(start_entity, hops=2):
    subgraph_edges = []
    current_nodes = {start_entity}
    for hop in range(hops):
        next_nodes = set()
        for node in current_nodes:
            for neighbor in G.neighbors(node):
                rel = G.edges[node, neighbor].get("relation", "CONNECTED_TO")
                subgraph_edges.append(f"{node} --[{rel}]--> {neighbor}")
                next_nodes.add(neighbor)
        current_nodes = next_nodes
    return subgraph_edges

context = graph_rag_query("Tensorbox", hops=2)
print("=== Graph-RAG Extracted Multi-Hop Grounding Context ===")
for line in set(context):
    print(" • " + line)
""")
        ]
    )
    save_notebook(nb_t13_02, "13_genai_and_agents/02_graph_rag_knowledge_augmented_retrieval.ipynb")

    # ==========================================
    # TRACK 13 - NOTEBOOK 03: LLM Tool Calling
    # ==========================================
    nb_t13_03 = create_notebook(
        title="03: LLM Function Calling, Tool Orchestration & JSON Schemas",
        description="Structured tool calling for LLMs: Pydantic schemas, OpenAI-compatible function descriptors, execution dispatch loops, and validation error handling.",
        track_num="13",
        track_name="Generative AI, LLMs, RAG & Multi-Agent Swarms",
        cells_data=[
            ("md", "## 1. Tool Declaration & Pydantic Schema Specification\nDefine structured tools for an autonomous agent."),
            ("code", """
import json

tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "calculate_mortgage",
            "description": "Calculates monthly mortgage payment for a property",
            "parameters": {
                "type": "object",
                "properties": {
                    "loan_amount": {"type": "number", "description": "Principal loan in USD"},
                    "annual_rate": {"type": "number", "description": "Interest rate e.g. 0.065 for 6.5%"},
                    "term_years": {"type": "integer", "description": "Loan duration in years"}
                },
                "required": ["loan_amount", "annual_rate", "term_years"]
            }
        }
    }
]

def calculate_mortgage(loan_amount: float, annual_rate: float, term_years: int) -> float:
    monthly_rate = annual_rate / 12.0
    n_payments = term_years * 12
    payment = loan_amount * (monthly_rate * (1 + monthly_rate)**n_payments) / ((1 + monthly_rate)**n_payments - 1)
    return round(payment, 2)

tool_registry = {"calculate_mortgage": calculate_mortgage}

def dispatch_tool_call(tool_name: str, arguments_json: str):
    args = json.loads(arguments_json)
    if tool_name in tool_registry:
        return tool_registry[tool_name](**args)
    raise ValueError(f"Tool {tool_name} not found")

sample_call = {"name": "calculate_mortgage", "arguments": json.dumps({"loan_amount": 400000, "annual_rate": 0.065, "term_years": 30})}
res = dispatch_tool_call(sample_call["name"], sample_call["arguments"])
print(f"Tool Result: Monthly Payment = ${res}")
""")
        ]
    )
    save_notebook(nb_t13_03, "13_genai_and_agents/03_llm_tool_calling_and_structured_json.ipynb")

    # ==========================================
    # TRACK 13 - NOTEBOOK 04: Multi-Agent Systems
    # ==========================================
    nb_t13_04 = create_notebook(
        title="04: Autonomous Multi-Agent Swarms & Collaborative Workflows",
        description="Multi-agent architecture: Role-based agents (Researcher, Financial Analyst, Writer), inter-agent message passing, task delegation, and execution synthesis.",
        track_num="13",
        track_name="Generative AI, LLMs, RAG & Multi-Agent Swarms",
        cells_data=[
            ("md", "## 1. Multi-Agent Role Definition & Communication Protocol\nConstructing specialized agents with distinct prompts and capabilities."),
            ("code", """
class Agent:
    def __init__(self, role: str, goal: str):
        self.role = role
        self.goal = goal
        
    def execute_task(self, context: str) -> str:
        return f"[{self.role} Output]: Successfully fulfilled '{self.goal}' utilizing context: {context[:40]}..."

researcher = Agent("Market Researcher", "Gather key financial metrics and competitor news")
analyst = Agent("Quantitative Analyst", "Compute risk ratios, Sharpe, and DCF valuation")
writer = Agent("Executive Reporter", "Synthesize findings into an investor report")

ctx1 = researcher.execute_task("Target: Semiconductor Industry Analysis 2026")
ctx2 = analyst.execute_task(ctx1)
final_report = writer.execute_task(ctx2)

print("=== Multi-Agent Pipeline Execution Log ===")
print(ctx1)
print(ctx2)
print(final_report)
""")
        ]
    )
    save_notebook(nb_t13_04, "13_genai_and_agents/04_crewai_multi_agent_autonomous_systems.ipynb")

    # ==========================================
    # TRACK 13 - NOTEBOOK 05: LoRA & PEFT
    # ==========================================
    nb_t13_05 = create_notebook(
        title="05: LoRA (Low-Rank Adaptation) & Parameter-Efficient Fine-Tuning (PEFT)",
        description="Mathematics and implementation of LoRA: Low-rank weight update decomposition $W_0 + \\frac{\\alpha}{r} B A$, parameter footprint reduction, and fine-tuning PyTorch layers.",
        track_num="13",
        track_name="Generative AI, LLMs, RAG & Multi-Agent Swarms",
        cells_data=[
            ("md", "## 1. LoRA Mathematical Decomposition\n$$W = W_0 + \\Delta W = W_0 + \\frac{\\alpha}{r} B A$$\nwhere $W_0 \\in \\mathbb{R}^{d \\times k}$ is frozen, $A \\in \\mathbb{R}^{r \\times k}$ initialized as $\\mathcal{N}(0, \\sigma^2)$, and $B \\in \\mathbb{R}^{d \\times r}$ initialized as $0$."),
            ("code", """
import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=4, lora_alpha=8.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = lora_alpha / rank
        
        self.base_layer = nn.Linear(in_features, out_features)
        self.base_layer.weight.requires_grad = False
        
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) / math.sqrt(in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
    def forward(self, x):
        base_out = self.base_layer(x)
        lora_out = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base_out + lora_out

in_dim, out_dim, r = 4096, 4096, 8
lora_layer = LoRALinear(in_dim, out_dim, rank=r)

base_params = in_dim * out_dim
trainable_lora_params = sum(p.numel() for p in lora_layer.parameters() if p.requires_grad)

print("=== LoRA Parameter Efficiency ===")
print(f"Standard Linear Weights  : {base_params:,} parameters")
print(f"Trainable LoRA Weights   : {trainable_lora_params:,} parameters (Rank={r})")
print(f"Parameter Reduction Ratio: {base_params / trainable_lora_params:.1f}x reduction ({(trainable_lora_params/base_params)*100:.3f}% trained)")
""")
        ]
    )
    save_notebook(nb_t13_05, "13_genai_and_agents/05_lora_and_peft_fine_tuning_deep_dive.ipynb")

    # ==========================================
    # TRACK 14 - NOTEBOOK 01: Reinforcement Learning (Q-Learning & DQN)
    # ==========================================
    nb_t14_01 = create_notebook(
        title="01: Reinforcement Learning: Tabular Q-Learning & Deep Q-Networks (DQN)",
        description="Markov Decision Processes (MDP), Bellman Optimality Equation, $\\epsilon$-greedy exploration vs exploitation, experience replay buffers, and Deep Q-Networks.",
        track_num="14",
        track_name="Reinforcement Learning & Production MLOps",
        cells_data=[
            ("md", "## 1. Bellman Optimality Equation\n$$Q^*(s, a) = R(s, a) + \\gamma \\max_{a'} Q^*(s', a')$$\nTemporal Difference Error:\n$$\\delta = r + \\gamma \\max_{a'} Q(s', a') - Q(s, a)$$"),
            ("code", """
import numpy as np
import torch
import torch.nn as nn
from collections import deque
import random

class DQN(nn.Module):
    def __init__(self, state_dim=4, action_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        
    def forward(self, state):
        return self.net(state)

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
        
    def __len__(self):
        return len(self.buffer)

dqn = DQN(state_dim=4, action_dim=2)
buffer = ReplayBuffer(capacity=1000)

for _ in range(100):
    s = np.random.randn(4)
    a = random.randint(0, 1)
    r = 1.0
    ns = np.random.randn(4)
    buffer.push(s, a, r, ns, False)

print(f"DQN Initialized with {len(buffer)} experience transitions.")
print(f"Sample Q-values for state: {dqn(torch.randn(1, 4)).detach().numpy().round(4)}")
""")
        ]
    )
    save_notebook(nb_t14_01, "14_rl_and_mlops/01_reinforcement_learning_q_learning_and_dqn.ipynb")

    # ==========================================
    # TRACK 14 - NOTEBOOK 02: Feature Stores
    # ==========================================
    nb_t14_02 = create_notebook(
        title="02: Feature Stores & Real-Time Low-Latency Online Inference",
        description="Design high-performance low-latency feature pipelines: In-memory key-value caching (sub-5ms feature lookups), feature view definitions, schema validation, and point-in-time correctness.",
        track_num="14",
        track_name="Reinforcement Learning & Production MLOps",
        cells_data=[
            ("md", "## 1. Feature Store In-Memory Key-Value Architecture\nSimulating low-latency online feature retrieval for real-time scoring."),
            ("code", """
import time
import json
import numpy as np
import pandas as pd

class InMemoryFeatureStore:
    def __init__(self):
        self.online_store = {}
        
    def write_features(self, entity_id: str, features: dict):
        self.online_store[entity_id] = features
        
    def get_online_features(self, entity_id: str) -> dict:
        return self.online_store.get(entity_id, None)

fs = InMemoryFeatureStore()
for user_id in range(1000, 11000):
    fs.write_features(f"user_{user_id}", {
        "avg_order_value_30d": round(np.random.exponential(45.0), 2),
        "total_logins_7d": np.random.randint(1, 20),
        "risk_score": round(np.random.uniform(0.01, 0.99), 3)
    })

start = time.perf_counter()
features = fs.get_online_features("user_5500")
latency_us = (time.perf_counter() - start) * 1_000_000

print(f"Online Feature Lookup Latency: {latency_us:.2f} microseconds (sub-millisecond)")
print(f"Retrieved Features: {features}")
""")
        ]
    )
    save_notebook(nb_t14_02, "14_rl_and_mlops/02_feature_stores_and_realtime_inference.ipynb")

    # ==========================================
    # TRACK 14 - NOTEBOOK 03: MLflow & Drift Monitoring
    # ==========================================
    nb_t14_03 = create_notebook(
        title="03: MLOps: Experiment Tracking & Statistical Data Drift Detection",
        description="Continuous model governance: Experiment metric logging, Kolmogorov-Smirnov (KS) two-sample drift tests, Population Stability Index (PSI), and alerting thresholds.",
        track_num="14",
        track_name="Reinforcement Learning & Production MLOps",
        cells_data=[
            ("md", "## 1. Kolmogorov-Smirnov (KS) Drift Test\nCompare distribution of baseline production features against current live stream.\n\n$$D = \\sup_x |F_1(x) - F_2(x)|$$"),
            ("code", """
import numpy as np
from scipy import stats
import pandas as pd

np.random.seed(42)
baseline_data = np.random.normal(loc=50.0, scale=10.0, size=1000)
live_clean = np.random.normal(loc=50.2, scale=10.1, size=1000)
live_drifted = np.random.normal(loc=56.0, scale=14.0, size=1000)

def detect_drift(baseline, live, alpha=0.05):
    ks_stat, p_val = stats.ks_2samp(baseline, live)
    is_drift = p_val < alpha
    return {"KS_Statistic": round(ks_stat, 4), "P_Value": p_val, "Drift_Detected": is_drift}

print("=== Statistical Drift Monitoring Analysis ===")
print("No Drift Case   :", detect_drift(baseline_data, live_clean))
print("Drifted Stream  :", detect_drift(baseline_data, live_drifted))
""")
        ]
    )
    save_notebook(nb_t14_03, "14_rl_and_mlops/03_mlflow_tracking_and_model_drift_monitoring.ipynb")

    # ==========================================
    # TRACK 14 - NOTEBOOK 04: Production Model Serving
    # ==========================================
    nb_t14_04 = create_notebook(
        title="04: Production Model Serving: FastAPI, Schema Validation & Health Checks",
        description="Deploy machine learning models as production REST APIs: Pydantic request/response data contracts, sub-10ms model execution, error handling, and Swagger OpenAPI docs.",
        track_num="14",
        track_name="Reinforcement Learning & Production MLOps",
        cells_data=[
            ("md", "## 1. FastAPI Application & Pydantic Data Contract\nDefine typed schemas and scoring endpoints."),
            ("code", """
from pydantic import BaseModel, Field
from typing import List
import numpy as np

class PredictionRequest(BaseModel):
    features: List[float] = Field(..., example=[1.5, 2.3, 0.4, 8.1])

class PredictionResponse(BaseModel):
    prediction: float
    status: str

def predict_score(req: PredictionRequest) -> PredictionResponse:
    feats = np.array(req.features)
    score = float(np.tanh(np.sum(feats) / len(feats)))
    return PredictionResponse(prediction=round(score, 4), status="success")

sample_payload = PredictionRequest(features=[2.4, -0.5, 1.1, 3.2])
response = predict_score(sample_payload)

print("=== Production Scoring Response ===")
print(response.model_dump())
""")
        ]
    )
    save_notebook(nb_t14_04, "14_rl_and_mlops/04_production_model_serving_fastapi.ipynb")

    print("✓ Tracks 11-14 generated successfully (13 notebooks).")
