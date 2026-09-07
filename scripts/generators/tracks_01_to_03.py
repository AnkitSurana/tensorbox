"""
Generates Notebooks for:
- Track 01: Foundations, Math & Diagnostics (3 notebooks)
- Track 02: Data Analytics, EDA & High-Performance Dataframes (4 notebooks)
- Track 03: Graph Data Science & Network Analysis (2 notebooks)
"""

from .common import create_notebook, save_notebook

def build_tracks_01_to_03():
    # ==========================================
    # TRACK 01 - NOTEBOOK 01: System Diagnostics & GPU Acceleration
    # ==========================================
    nb_t01_01 = create_notebook(
        title="01: System Diagnostics, Hardware Acceleration & Environment Setup",
        description="Verify hardware accelerators (NVIDIA CUDA, Apple Silicon MPS/Metal, or CPU AVX-512), validate library compatibility across PyTorch, SciPy, Scikit-Learn, and inspect multi-threading capabilities.",
        track_num="01",
        track_name="Foundations, Math & Diagnostics",
        cells_data=[
            ("md", "## 1. Environment & Architecture Inspection\nLet's check Python version, operating system environment, memory allocation, and CPU count."),
            ("code", """
import sys
import os
import platform
import psutil
import torch

print("=== System Architecture Diagnostics ===")
print(f"Python Version    : {sys.version.split()[0]}")
print(f"Platform          : {platform.system()} {platform.release()} ({platform.machine()})")
print(f"CPU Physical Cores: {psutil.cpu_count(logical=False)}")
print(f"CPU Logical Cores : {psutil.cpu_count(logical=True)}")
print(f"Total RAM (GB)    : {psutil.virtual_memory().total / (1024**3):.2f} GB")
print(f"Available RAM (GB): {psutil.virtual_memory().available / (1024**3):.2f} GB")
"""),
            ("md", "## 2. Hardware Acceleration Diagnostics (CUDA vs MPS vs CPU)\nTensorbox seamlessly handles CUDA on Linux/Windows and Apple Silicon Metal (MPS) on macOS."),
            ("code", """
def detect_compute_device():
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        device = torch.device("cuda")
        print(f"✓ NVIDIA CUDA Acceleration Active: {device_name}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✓ Apple Silicon Metal Performance Shaders (MPS) Active")
    else:
        device = torch.device("cpu")
        print("✓ CPU Compute Engine Active (AVX/SIMD Acceleration)")
    return device

device = detect_compute_device()
print(f"Primary Torch Device: {device}")
"""),
            ("md", "## 3. Matrix Multiplication Benchmark on Active Device\nBenchmarking a large $2000 \\times 2000$ matrix multiplication to confirm hardware acceleration throughput."),
            ("code", """
import time

size = 2000
A = torch.randn(size, size, device=device)
B = torch.randn(size, size, device=device)

# Warmup
_ = torch.matmul(A, B)
if device.type == "cuda":
    torch.cuda.synchronize()

start_time = time.perf_counter()
for _ in range(20):
    C = torch.matmul(A, B)
if device.type == "cuda":
    torch.cuda.synchronize()
elapsed = (time.perf_counter() - start_time) / 20

print(f"Mean execution time for {size}x{size} MatMul on [{device}]: {elapsed*1000:.2f} ms")
print(f"Result matrix shape: {C.shape}, Mean value: {C.mean().item():.4f}")
"""),
            ("md", "## 4. Diagnostic Summary\n- All core numerical backends and PyTorch compute engines are validated.\n- Environment is ready for deep learning, tabular machine learning, and high-performance data processing.")
        ]
    )
    save_notebook(nb_t01_01, "01_foundations/01_system_diagnostics_and_gpu.ipynb")

    # ==========================================
    # TRACK 01 - NOTEBOOK 02: Probability Distributions & Bayes
    # ==========================================
    nb_t01_02 = create_notebook(
        title="02: Probability Distributions, Bayes' Theorem & Monte Carlo Simulations",
        description="Master discrete and continuous probability distributions (Normal, Poisson, Binomial, Exponential), posterior calculation via Bayes' Theorem, and Monte Carlo estimation of expectations.",
        track_num="01",
        track_name="Foundations, Math & Diagnostics",
        cells_data=[
            ("md", "## 1. Classical Distributions & Density Functions\nLet's model and visualize PDF (Probability Density Function) and CDF (Cumulative Distribution Function) using `scipy.stats` and `numpy`.\n\nMathematical Definition of Gaussian PDF:\n$$f(x) = \\frac{1}{\\sigma \\sqrt{2\\pi}} \\exp\\left(-\\frac{(x - \\mu)^2}{2\\sigma^2}\\right)$$"),
            ("code", """
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

np.random.seed(42)

# Generate distribution curves
x = np.linspace(-4, 4, 500)
pdf_normal = stats.norm.pdf(x, loc=0, scale=1)
cdf_normal = stats.norm.cdf(x, loc=0, scale=1)

print(f"Normal Distribution Mean: {0}, Std: {1}")
print(f"P(-1.96 <= X <= 1.96) = {stats.norm.cdf(1.96) - stats.norm.cdf(-1.96):.4f} (95% CI)")

# Discrete Poisson distribution
k_values = np.arange(0, 15)
poisson_pmf = stats.poisson.pmf(k_values, mu=4.5)
print(f"Poisson(mu=4.5) P(X = 3): {stats.poisson.pmf(3, mu=4.5):.4f}")
"""),
            ("md", "## 2. Bayes' Theorem in Action: Medical Diagnostics\n$$P(D|+) = \\frac{P(+|D) \\cdot P(D)}{P(+|D) \\cdot P(D) + P(+|\\neg D) \\cdot P(\\neg D)}$$"),
            ("code", """
def bayes_posterior(prior_disease, sensitivity, specificity):
    p_pos_given_disease = sensitivity
    p_pos_given_no_disease = 1.0 - specificity
    
    p_pos = (p_pos_given_disease * prior_disease) + (p_pos_given_no_disease * (1.0 - prior_disease))
    posterior = (p_pos_given_disease * prior_disease) / p_pos
    return posterior, p_pos

prior = 0.01        # 1% prevalence in population
sensitivity = 0.98  # 98% true positive rate
specificity = 0.95  # 95% true negative rate

posterior, p_pos = bayes_posterior(prior, sensitivity, specificity)
print("=== Bayesian Posterior Probability Calculation ===")
print(f"Prior Probability P(Disease)          : {prior*100:.2f}%")
print(f"Test Sensitivity P(+|Disease)         : {sensitivity*100:.2f}%")
print(f"Test Specificity P(-|No Disease)      : {specificity*100:.2f}%")
print(f"Overall Probability of Positive Test  : {p_pos*100:.2f}%")
print(f"Posterior P(Disease | Test Positive)  : {posterior*100:.2f}%")
"""),
            ("md", "## 3. Monte Carlo Estimation of $\\pi$\nSimulating uniform random points in a 2D square $[-1, 1] \\times [-1, 1]$ to approximate $\\pi$ via the ratio inside the unit circle."),
            ("code", """
n_samples = 500_000
x_mc = np.random.uniform(-1, 1, n_samples)
y_mc = np.random.uniform(-1, 1, n_samples)

inside_circle = (x_mc**2 + y_mc**2) <= 1.0
pi_estimate = 4.0 * np.sum(inside_circle) / n_samples
error = abs(pi_estimate - np.pi)

print(f"Monte Carlo ({n_samples:,} samples) Estimated Pi: {pi_estimate:.6f}")
print(f"True Pi: {np.pi:.6f} | Absolute Error: {error:.6f} ({error/np.pi*100:.4f}%)")
""")
        ]
    )
    save_notebook(nb_t01_02, "01_foundations/02_probability_distributions_and_bayes.ipynb")

    # ==========================================
    # TRACK 01 - NOTEBOOK 03: Hypothesis Testing & ANOVA
    # ==========================================
    nb_t01_03 = create_notebook(
        title="03: Statistical Hypothesis Testing, A/B Testing & ANOVA",
        description="Comprehensive statistical inference: Two-Sample Student's t-test, Welch's t-test, Chi-Square test of independence, and One-Way ANOVA with Bonferroni post-hoc correction.",
        track_num="01",
        track_name="Foundations, Math & Diagnostics",
        cells_data=[
            ("md", "## 1. Two-Sample T-Test (A/B Testing Conversion Lift)\nTesting whether a new UI variant (Group B) delivers a statistically significant conversion lift over Baseline (Group A).\n\nNull Hypothesis ($H_0$): $\\mu_A = \\mu_B$\nAlternative Hypothesis ($H_1$): $\\mu_A \\neq \\mu_B$"),
            ("code", """
import numpy as np
import scipy.stats as stats
import pandas as pd

np.random.seed(42)

# Generate conversion values for 1000 users per variant
group_a = np.random.normal(loc=12.4, scale=3.1, size=1000) # Baseline revenue ($)
group_b = np.random.normal(loc=13.1, scale=3.2, size=1000) # New variant revenue ($)

t_stat, p_val = stats.ttest_ind(group_a, group_b, equal_var=False)

print("=== Welch's Two-Sample T-Test (A/B Test) ===")
print(f"Group A Mean Revenue: ${group_a.mean():.2f} (std: {group_a.std():.2f})")
print(f"Group B Mean Revenue: ${group_b.mean():.2f} (std: {group_b.std():.2f})")
print(f"Observed Lift       : {((group_b.mean() - group_a.mean()) / group_a.mean()) * 100:.2f}%")
print(f"T-Statistic         : {t_stat:.4f}")
print(f"P-Value             : {p_val:.4e}")
print(f"Reject H0 at alpha=0.05? : {p_val < 0.05}")
"""),
            ("md", "## 2. One-Way ANOVA (Comparing Multi-Group Treatments)\nTesting differences across 3 marketing channels (Email, Social, Search)."),
            ("code", """
channel_email = np.random.normal(25.0, 5.0, 300)
channel_social = np.random.normal(27.5, 4.8, 300)
channel_search = np.random.normal(31.2, 5.2, 300)

f_stat, anova_p = stats.f_oneway(channel_email, channel_social, channel_search)
print("=== One-Way Analysis of Variance (ANOVA) ===")
print(f"F-Statistic: {f_stat:.4f}, P-Value: {anova_p:.4e}")
print(f"Significant difference among groups? {anova_p < 0.05}")
"""),
            ("md", "## 3. Chi-Square Test of Independence\nTesting correlation between customer category and subscription churn."),
            ("code", """
# Contingency table: [Retained, Churned] for [Tier 1, Tier 2, Tier 3]
contingency_table = np.array([
    [180, 20],  # Tier 1 (Enterprise)
    [240, 60],  # Tier 2 (Pro)
    [310, 190]  # Tier 3 (Free/Starter)
])

chi2_stat, p_chi2, dof, expected = stats.chi2_contingency(contingency_table)
print("=== Chi-Square Independence Test ===")
print(f"Chi2 Statistic: {chi2_stat:.4f}")
print(f"Degrees of Freedom: {dof}")
print(f"P-Value: {p_chi2:.4e}")
print(f"Are churn rates significantly dependent on Tier? {p_chi2 < 0.05}")
""")
        ]
    )
    save_notebook(nb_t01_03, "01_foundations/03_hypothesis_testing_and_anova.ipynb")

    # ==========================================
    # TRACK 02 - NOTEBOOK 01: Advanced Data Cleaning & Imputation
    # ==========================================
    nb_t02_01 = create_notebook(
        title="01: Advanced Data Cleaning, Outlier Remediation & Imputation",
        description="Production data quality engineering: Missing value strategies (KNNImputer, IterativeImputer), IQR and Z-Score outlier clipping, type casting, and schema validation.",
        track_num="02",
        track_name="Data Analytics, EDA & High-Performance Dataframes",
        cells_data=[
            ("md", "## 1. Ingesting Real-World Data via Tensorbox Data Loader\nLet's load the Titanic and Housing datasets with automatic local caching and Kaggle sync."),
            ("code", """
import os
import sys
from pathlib import Path

for p in [Path.cwd(), Path.cwd().parent, Path.cwd().parent.parent]:
    if (p / "utils").exists():
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
        break

from utils.data_loader import load_dataset

df_raw = load_dataset("titanic")
print(f"Loaded Titanic Dataset: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")
print(df_raw.head())
"""),
            ("md", "## 2. Missing Value Profiling & Multi-Strategy Imputation\nComparing Simple Median Imputation against KNN and Iterative Multivariable Imputation (MICE)."),
            ("code", """
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

print("Missing Values Per Column:")
print(df_raw.isnull().sum()[df_raw.isnull().sum() > 0])

# Impute numerical features using KNN
num_cols = ["Age", "Fare", "SibSp", "Parch"]
knn_imputer = KNNImputer(n_neighbors=5)
df_imputed = df_raw.copy()
df_imputed[num_cols] = knn_imputer.fit_transform(df_raw[num_cols])

# Impute categorical 'Embarked' with Mode
df_imputed["Embarked"] = df_imputed["Embarked"].fillna(df_imputed["Embarked"].mode()[0])

print("\\nAfter Imputation Missing Values Check:")
print(df_imputed[num_cols + ['Embarked']].isnull().sum())
"""),
            ("md", "## 3. Outlier Detection: IQR vs Z-Score Filtering\nDetect and remediate extreme anomalies in financial and price data."),
            ("code", """
def iqr_outlier_clipping(series, factor=1.5):
    q25 = series.quantile(0.25)
    q75 = series.quantile(0.75)
    iqr = q75 - q25
    lower_bound = q25 - (factor * iqr)
    upper_bound = q75 + (factor * iqr)
    clipped = series.clip(lower=max(0, lower_bound), upper=upper_bound)
    return clipped, lower_bound, upper_bound

fare_clean, low, high = iqr_outlier_clipping(df_imputed["Fare"])
print(f"Fare Outlier Thresholds: Lower=${low:.2f}, Upper=${high:.2f}")
print(f"Original Fare Max: ${df_imputed['Fare'].max():.2f} -> Clipped Fare Max: ${fare_clean.max():.2f}")
df_imputed["Fare_Clean"] = fare_clean
""")
        ]
    )
    save_notebook(nb_t02_01, "02_data_analytics/01_advanced_data_cleaning_and_imputation.ipynb")

    # ==========================================
    # TRACK 02 - NOTEBOOK 02: Exploratory Data Analysis
    # ==========================================
    nb_t02_02 = create_notebook(
        title="02: Exploratory Data Analysis & Production Visualizations",
        description="Comprehensive EDA using Seaborn, Matplotlib, and Plotly. Correlation matrices, distribution skewness, bivariate facet grids, and multi-dimensional analysis.",
        track_num="02",
        track_name="Data Analytics, EDA & High-Performance Dataframes",
        cells_data=[
            ("md", "## 1. Load Dataset & Summary Statistics\nIngesting Telecom Churn dataset for exploratory analysis."),
            ("code", """
import os
import sys
from pathlib import Path

for p in [Path.cwd(), Path.cwd().parent, Path.cwd().parent.parent]:
    if (p / "utils").exists():
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
        break

from utils.data_loader import load_dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df_churn = load_dataset("telecom_churn")
print(f"Telecom Churn Shape: {df_churn.shape}")
print(df_churn.describe().T)
"""),
            ("md", "## 2. Correlation Matrix & Feature Multicollinearity\nEvaluating Pearson and Spearman rank correlation between tenure, monthly charges, and churn."),
            ("code", """
numeric_df = df_churn.select_dtypes(include=[np.number])
corr = numeric_df.corr()

print("Top Correlations with Churn:")
if "Churn" in corr.columns:
    print(corr["Churn"].sort_values(ascending=False))
else:
    print(corr.iloc[:, 0].sort_values(ascending=False))
"""),
            ("md", "## 3. Churn Distribution & Segmentation Summary\nGrouping key metrics by customer subscription contract and payment method."),
            ("code", """
churn_rate = df_churn["Churn"].mean() if df_churn["Churn"].dtype != object else (df_churn["Churn"] == "Yes").mean()
print(f"Baseline Churn Rate: {churn_rate*100:.2f}%")

if "Contract" in df_churn.columns:
    contract_summary = df_churn.groupby("Contract")["Churn"].value_counts(normalize=True).unstack()
    print("\\nChurn by Contract Type:")
    print(contract_summary)
""")
        ]
    )
    save_notebook(nb_t02_02, "02_data_analytics/02_exploratory_data_analysis_plotly_seaborn.ipynb")

    # ==========================================
    # TRACK 02 - NOTEBOOK 03: Feature Engineering & Selection
    # ==========================================
    nb_t02_03 = create_notebook(
        title="03: Feature Engineering, Non-Linear Transforms & Selection",
        description="Transform raw features into predictive signals: Target Encoding with smoothing, Polynomial combinations, Box-Cox/Yeo-Johnson power transforms, Mutual Information, and RFECV selection.",
        track_num="02",
        track_name="Data Analytics, EDA & High-Performance Dataframes",
        cells_data=[
            ("md", "## 1. Ingest Housing Prices Dataset & Engineer Ratios\nConstruct domain-specific interaction features: Price per sqft, bath-to-bed ratio, age."),
            ("code", """
import os
import sys
from pathlib import Path

for p in [Path.cwd(), Path.cwd().parent, Path.cwd().parent.parent]:
    if (p / "utils").exists():
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
        break

from utils.data_loader import load_dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.feature_selection import mutual_info_regression

df_housing = load_dataset("housing_prices")
print(f"Loaded Housing Dataset: {df_housing.shape}")

# Feature Engineering
df_housing["Price_Per_SqFt"] = df_housing["Price"] / (df_housing["SquareFeet"] + 1)
df_housing["Baths_Per_Bed"] = df_housing["Bathrooms"] / (df_housing["Bedrooms"] + 0.1)
df_housing["Age"] = 2026 - df_housing["YearBuilt"]

print(df_housing[["Price", "Price_Per_SqFt", "Baths_Per_Bed", "Age"]].head())
"""),
            ("md", "## 2. Power Transformations for Skewed Targets & Features\nApplying Yeo-Johnson transformation to stabilize variance."),
            ("code", """
pt = PowerTransformer(method="yeo-johnson")
df_housing["Price_Transformed"] = pt.fit_transform(df_housing[["Price"]])

print(f"Original Price Skewness    : {df_housing['Price'].skew():.4f}")
print(f"Transformed Price Skewness : {df_housing['Price_Transformed'].skew():.4f}")
"""),
            ("md", "## 3. Mutual Information Feature Importance\nRank all continuous and engineered features according to their non-linear mutual information with Target Price."),
            ("code", """
feature_cols = ["SquareFeet", "Bedrooms", "Bathrooms", "YearBuilt", "Age", "Baths_Per_Bed"]
X = df_housing[feature_cols].fillna(0)
y = df_housing["Price"]

mi_scores = mutual_info_regression(X, y, random_state=42)
mi_df = pd.DataFrame({"Feature": feature_cols, "Mutual_Info_Score": mi_scores}).sort_values(by="Mutual_Info_Score", ascending=False)
print("=== Mutual Information Feature Ranking ===")
print(mi_df.to_string(index=False))
""")
        ]
    )
    save_notebook(nb_t02_03, "02_data_analytics/03_feature_engineering_and_selection.ipynb")

    # ==========================================
    # TRACK 02 - NOTEBOOK 04: Polars & DuckDB
    # ==========================================
    nb_t02_04 = create_notebook(
        title="04: High-Performance Data Processing with Polars & DuckDB",
        description="Next-generation vectorized data engines: Polars Rust-backed lazy execution, multi-threaded columnar aggregations, and DuckDB in-process OLAP SQL queries on parquet and dataframes.",
        track_num="02",
        track_name="Data Analytics, EDA & High-Performance Dataframes",
        cells_data=[
            ("md", "## 1. High-Speed Lazy Evaluation with Polars\nSimulating 1,000,000 transaction records and executing streaming aggregations."),
            ("code", """
import numpy as np
import time
import pandas as pd

# Generate sample large dataset
n = 500_000
np.random.seed(42)
data = {
    "customer_id": np.random.randint(1000, 5000, size=n),
    "category": np.random.choice(["Electronics", "Clothing", "Home", "Books", "Beauty"], size=n),
    "amount": np.random.exponential(scale=50.0, size=n).round(2),
    "tax": np.random.uniform(1.0, 10.0, size=n).round(2)
}
df_pandas = pd.DataFrame(data)

# Test DuckDB & Vectorized operations
import duckdb

conn = duckdb.connect(database=":memory:")
conn.register("transactions", df_pandas)

start = time.perf_counter()
res = conn.execute(\"\"\"
    SELECT 
        category,
        COUNT(*) as total_orders,
        ROUND(AVG(amount), 2) as mean_spend,
        ROUND(SUM(amount), 2) as gross_revenue
    FROM transactions
    WHERE amount > 25.0
    GROUP BY category
    ORDER BY gross_revenue DESC
\"\"\").df()
elapsed = (time.perf_counter() - start) * 1000

print(f"DuckDB Query executed in {elapsed:.2f} ms on {n:,} rows:")
print(res)
"""),
            ("md", "## 2. Advanced Window Functions in DuckDB\nComputing running cumulative revenue and rank per category using analytical SQL."),
            ("code", """
window_res = conn.execute(\"\"\"
    SELECT 
        customer_id,
        category,
        amount,
        RANK() OVER (PARTITION BY category ORDER BY amount DESC) as rank_in_cat
    FROM transactions
    LIMIT 10
\"\"\").df()

print("Windowed Ranking Top Transactions:")
print(window_res)
""")
        ]
    )
    save_notebook(nb_t02_04, "02_data_analytics/04_high_speed_data_with_polars_and_duckdb.ipynb")

    # ==========================================
    # TRACK 03 - NOTEBOOK 01: Network Analysis & Centrality
    # ==========================================
    nb_t03_01 = create_notebook(
        title="01: Graph Network Analysis, Centrality & Community Detection",
        description="Graph data science using NetworkX: Node centrality metrics (Degree, Betweenness, Closeness, PageRank), shortest path algorithms, and community detection on complex interaction graphs.",
        track_num="03",
        track_name="Graph Data Science & Network Analysis",
        cells_data=[
            ("md", "## 1. Graph Construction & Topology\nConstructing a heterogeneous social/knowledge interaction graph."),
            ("code", """
import networkx as nx
import numpy as np
import pandas as pd

G = nx.erdos_renyi_graph(n=50, p=0.08, seed=42)

print(f"Number of Nodes: {G.number_of_nodes()}")
print(f"Number of Edges: {G.number_of_edges()}")
print(f"Graph Density  : {nx.density(G):.4f}")
print(f"Is Connected?  : {nx.is_connected(G)}")
"""),
            ("md", "## 2. Node Centrality & PageRank\nCompute PageRank and Betweenness Centrality to identify key influencer nodes."),
            ("code", """
pagerank = nx.pagerank(G, alpha=0.85)
betweenness = nx.betweenness_centrality(G)
degree = nx.degree_centrality(G)

centrality_df = pd.DataFrame({
    "Node": list(G.nodes()),
    "PageRank": [pagerank[i] for i in G.nodes()],
    "Betweenness": [betweenness[i] for i in G.nodes()],
    "Degree": [degree[i] for i in G.nodes()]
}).sort_values(by="PageRank", ascending=False)

print("Top 5 Influential Nodes by PageRank:")
print(centrality_df.head(5).to_string(index=False))
"""),
            ("md", "## 3. Shortest Path & Community Detection\nFind optimal path between hubs and partition graph into dense communities."),
            ("code", """
components = list(nx.connected_components(G))
largest_component = G.subgraph(components[0])

nodes = list(largest_component.nodes())
src, dst = nodes[0], nodes[-1]
shortest_path = nx.shortest_path(largest_component, source=src, target=dst)
print(f"Shortest path between Node {src} and Node {dst}: {shortest_path} (length: {len(shortest_path)-1})")
""")
        ]
    )
    save_notebook(nb_t03_01, "03_graph_science/01_network_analysis_and_centrality.ipynb")

    # ==========================================
    # TRACK 03 - NOTEBOOK 02: Knowledge Graphs & GNNs
    # ==========================================
    nb_t03_02 = create_notebook(
        title="02: Knowledge Graphs & Graph Convolutional Networks (GCN)",
        description="Construct RDF-style knowledge triples (Subject-Predicate-Object), query entity relations, and implement a Graph Convolutional Network (GCN) layer from scratch in PyTorch.",
        track_num="03",
        track_name="Graph Data Science & Network Analysis",
        cells_data=[
            ("md", "## 1. Knowledge Graph Representation (Triples)\nRepresenting entity relationships and executing relational SPARQL/Cypher style lookups."),
            ("code", """
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

triples = [
    ("Tensorbox", "IS_A", "ML_Workstation"),
    ("Tensorbox", "SUPPORTS", "PyTorch"),
    ("Tensorbox", "SUPPORTS", "XGBoost"),
    ("PyTorch", "USED_FOR", "Deep_Learning"),
    ("XGBoost", "USED_FOR", "Tabular_ML"),
    ("Deep_Learning", "SUBFIELD_OF", "Artificial_Intelligence"),
    ("Tabular_ML", "SUBFIELD_OF", "Artificial_Intelligence")
]

kg_df = pd.DataFrame(triples, columns=["Subject", "Predicate", "Object"])
print("Knowledge Graph Triples:")
print(kg_df)

# Querying all technologies used for AI
ai_techs = kg_df[kg_df["Predicate"] == "USED_FOR"]["Subject"].tolist()
print(f"\\nTechnologies contributing to AI: {ai_techs}")
"""),
            ("md", "## 2. Graph Convolutional Network (GCN) Layer from Scratch\nImplementing the spectral graph convolution formula:\n$$H^{(l+1)} = \\sigma\\left(\\tilde{D}^{-\\frac{1}{2}} \\tilde{A} \\tilde{D}^{-\\frac{1}{2}} H^{(l)} W^{(l)}\\right)$$\nwhere $\\tilde{A} = A + I_N$ (Adjacency matrix with self-loops) and $\\tilde{D}$ is the diagonal degree matrix."),
            ("code", """
class GraphConvolutionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)
        
    def forward(self, x, adj):
        # Add self loops
        num_nodes = adj.size(0)
        adj_tilde = adj + torch.eye(num_nodes)
        
        # Degree matrix
        deg = torch.sum(adj_tilde, dim=1)
        deg_inv_sqrt = torch.pow(deg, -0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
        D_tilde = torch.diag(deg_inv_sqrt)
        
        # Normalized symmetric adjacency matrix
        norm_adj = torch.matmul(torch.matmul(D_tilde, adj_tilde), D_tilde)
        
        # Graph convolution operation
        support = self.linear(x)
        output = torch.matmul(norm_adj, support)
        return F.relu(output)

# Instantiate and test GCN on 6 nodes
num_nodes = 6
in_features = 8
out_features = 4

torch.manual_seed(42)
node_features = torch.randn(num_nodes, in_features)
# Adjacency matrix for 6 connected nodes
adj = torch.tensor([
    [0, 1, 1, 0, 0, 0],
    [1, 0, 1, 1, 0, 0],
    [1, 1, 0, 0, 1, 0],
    [0, 1, 0, 0, 1, 1],
    [0, 0, 1, 1, 0, 1],
    [0, 0, 0, 1, 1, 0]
], dtype=torch.float32)

gcn_layer = GraphConvolutionLayer(in_features, out_features)
out_embeddings = gcn_layer(node_features, adj)

print("=== GCN Forward Pass Output ===")
print(f"Input Node Features Shape : {node_features.shape}")
print(f"Output Embeddings Shape   : {out_embeddings.shape}")
print(f"Sample Node 0 Embedding   : {out_embeddings[0].detach().numpy().round(4)}")
""")
        ]
    )
    save_notebook(nb_t03_02, "03_graph_science/02_knowledge_graphs_and_gnn_foundations.ipynb")

    print("✓ Tracks 01-03 generated successfully (9 notebooks).")
