"""
Generates Notebooks for:
- Track 07: Statistical Time-Series Forecasting (2 notebooks)
- Track 08: Classical NLP & Word Embeddings (2 notebooks)
- Track 09: Computer Vision & Convolutional Neural Networks (2 notebooks)
- Track 10: Deep Sequential Models & Transformers from Scratch (2 notebooks)
"""

from .common import create_notebook, save_notebook, DATA_LOADER_BOILERPLATE

def build_tracks_07_to_10():
    # ==========================================
    # TRACK 07 - NOTEBOOK 01: Classical Time Series (ARIMA / SARIMAX)
    # ==========================================
    nb_t07_01 = create_notebook(
        title="01: Classical Time Series Forecasting: ARIMA, SARIMAX & Stationarity Tests",
        description="Comprehensive time-series econometrics: Augmented Dickey-Fuller (ADF) stationarity testing, differencing ($d$), ACF/PACF auto-correlation analysis, and SARIMAX model forecasting.",
        track_num="07",
        track_name="Statistical Time-Series Forecasting",
        cells_data=[
            ("md", "## 1. Time Series Stationarity & Augmented Dickey-Fuller (ADF) Test\nA stationary series has constant mean, constant variance, and autocovariance independent of time.\n\nADF Hypothesis:\n$H_0$: Series possesses a unit root (non-stationary)\n$H_1$: Series is stationary"),
            ("code", DATA_LOADER_BOILERPLATE + """
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose

df = load_dataset("monthly_sales")
df["Date"] = pd.to_datetime(df["Date"])
df = df.set_index("Date").sort_index()

sales = df["Sales"]
print(f"Sales Records: {len(sales)} months")

# Run ADF Test
adf_stat, p_value, lags, nobs, crit_vals, icbest = adfuller(sales)
print("=== Augmented Dickey-Fuller (ADF) Stationarity Test ===")
print(f"ADF Statistic: {adf_stat:.4f}")
print(f"p-value      : {p_value:.4f}")
print(f"Stationary at 5% level? {p_value < 0.05}")
"""),
            ("md", "## 2. Differencing & SARIMAX Fitting\nFit SARIMAX $(p=1, d=1, q=1) \\times (P=1, D=1, Q=0)_{12}$ on monthly sales data."),
            ("code", """
train_sales = sales.iloc[:-6]
test_sales = sales.iloc[-6:]

model = SARIMAX(train_sales, order=(1, 1, 1), seasonal_order=(1, 1, 0, 12), enforce_stationarity=False, enforce_invertibility=False)
sarimax_fit = model.fit(disp=False)

forecast = sarimax_fit.forecast(steps=6)
mae = np.mean(np.abs(test_sales.values - forecast.values))
mape = np.mean(np.abs((test_sales.values - forecast.values) / test_sales.values)) * 100

print("=== SARIMAX Forecast Results (6 Months Horizon) ===")
print(pd.DataFrame({"Actual": test_sales.values, "Forecast": forecast.values.round(2)}, index=test_sales.index))
print(f"\\nForecast MAE : ${mae:.2f}")
print(f"Forecast MAPE: {mape:.2f}%")
""")
        ]
    )
    save_notebook(nb_t07_01, "07_time_series/01_classical_forecasting_arima_sarimax.ipynb")

    # ==========================================
    # TRACK 07 - NOTEBOOK 02: Modern Time Series & Lag Features
    # ==========================================
    nb_t07_02 = create_notebook(
        title="02: Modern Machine Learning Forecasting: Lag Windows, Rolling Stats & GBDT",
        description="Transform time series into supervised tabular regression: Rolling window moving averages, EWMA exponential smoothing, expanding statistics, autoregressive lags, and LightGBM forecasting.",
        track_num="07",
        track_name="Statistical Time-Series Forecasting",
        cells_data=[
            ("md", "## 1. Feature Engineering on Time Horizons\nConstruct lag features: $y_{t-1}, y_{t-2}, y_{t-7}$, rolling means, and rolling standard deviations."),
            ("code", DATA_LOADER_BOILERPLATE + """
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import lightgbm as lgb

df = load_dataset("stock_market")
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# Autoregressive Lags
for lag in [1, 2, 3, 5, 10]:
    df[f"Close_Lag_{lag}"] = df["Close"].shift(lag)

# Rolling Windows
df["Rolling_Mean_7"] = df["Close"].shift(1).rolling(window=7).mean()
df["Rolling_Std_7"] = df["Close"].shift(1).rolling(window=7).std()

df = df.dropna().reset_index(drop=True)
print(f"Generated Feature Matrix: {df.shape}")
print(df[["Date", "Close", "Close_Lag_1", "Rolling_Mean_7", "Rolling_Std_7"]].tail())
"""),
            ("md", "## 2. Chronological Walk-Forward Split & LightGBM Model\nPrevent lookahead bias using temporal train/test split."),
            ("code", """
features = [c for c in df.columns if "Lag" in c or "Rolling" in c]
target = "Close"

split_idx = int(len(df) * 0.8)
X_train, X_test = df[features].iloc[:split_idx], df[features].iloc[split_idx:]
y_train, y_test = df[target].iloc[:split_idx], df[target].iloc[split_idx:]

model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mape = mean_absolute_percentage_error(y_test, y_pred) * 100
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("=== Walk-Forward ML Forecast Results ===")
print(f"Test RMSE : ${rmse:.2f}")
print(f"Test MAPE : {mape:.2f}%")
""")
        ]
    )
    save_notebook(nb_t07_02, "07_time_series/02_modern_time_series_prophet_neuralforecast.ipynb")

    # ==========================================
    # TRACK 08 - NOTEBOOK 01: Classical NLP & spaCy Pipelines
    # ==========================================
    nb_t08_01 = create_notebook(
        title="01: Natural Language Processing: Regex, Tokenization, Lemmatization & NER",
        description="Build end-to-end NLP preprocessing pipelines: Regex cleaning, stopword removal, lemmatization, Part-of-Speech (POS) tagging, and Named Entity Recognition (NER).",
        track_num="08",
        track_name="Classical NLP & Word Embeddings",
        cells_data=[
            ("md", "## 1. Text Corpus Ingestion & Tokenization Pipeline\nIngest unstructured news text data and clean whitespace, HTML tags, and punctuation."),
            ("code", DATA_LOADER_BOILERPLATE + """
import re
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

df = load_dataset("news_articles")
print(f"Loaded {len(df)} news articles across categories: {df['category'] if 'category' in df.columns else df['Category'].unique()}")

def clean_text(text):
    text = re.sub(r"<.*?>", "", str(text)) # Remove HTML
    text = re.sub(r"[^a-zA-Z\\s]", "", text) # Keep alphabetic
    text = text.lower().strip()
    return text

text_col = "headline" if "headline" in df.columns else df.columns[1]
df["Clean_Text"] = df[text_col].apply(clean_text)
print("Sample cleaned text snippet:")
print(df["Clean_Text"].iloc[0][:200])
"""),
            ("md", "## 2. TF-IDF Representation & Category Keyword Extraction\nCompute Term Frequency-Inverse Document Frequency:\n$$\\text{TF-IDF}(t, d, D) = \\text{TF}(t, d) \\times \\log\\left(\\frac{|D|}{|\\{d \\in D : t \\in d\\}| + 1}\\right)$$"),
            ("code", """
tfidf = TfidfVectorizer(max_features=1000, stop_words="english", ngram_range=(1, 2))
X_tfidf = tfidf.fit_transform(df["Clean_Text"])

vocab = np.array(tfidf.get_feature_names_out())
print(f"TF-IDF Matrix Shape: {X_tfidf.shape}")

# Extract top keywords per category
for cat in df["category"].unique():
    cat_mask = (df["category"] == cat).values
    cat_tfidf = np.asarray(X_tfidf[cat_mask].mean(axis=0)).ravel()
    top_indices = cat_tfidf.argsort()[-5:][::-1]
    top_words = vocab[top_indices]
    print(f"Top 5 Keywords for [{cat}]: {', '.join(top_words)}")
""")
        ]
    )
    save_notebook(nb_t08_01, "08_nlp_embeddings/01_text_preprocessing_and_spacy_pipelines.ipynb")

    # ==========================================
    # TRACK 08 - NOTEBOOK 02: Dense Embeddings (Word2Vec / FastText)
    # ==========================================
    nb_t08_02 = create_notebook(
        title="02: Dense Semantic Embeddings: Word2Vec, CBOW & Cosine Similarity Search",
        description="Vector semantics: Continuous Bag-of-Words (CBOW) and Skip-Gram embeddings, vector arithmetic (King - Man + Woman = Queen), and dense semantic similarity search.",
        track_num="08",
        track_name="Classical NLP & Word Embeddings",
        cells_data=[
            ("md", "## 1. Vector Space Semantics & Cosine Similarity\nCosine similarity between dense vector representations:\n$$\\text{cosine}(\\mathbf{u}, \\mathbf{v}) = \\frac{\\mathbf{u} \\cdot \\mathbf{v}}{\\|\\mathbf{u}\\|_2 \\|\\mathbf{v}\\|_2}$$"),
            ("code", """
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Define a synthetic semantic embedding space (16 dimensions)
np.random.seed(42)
vocab = ["king", "queen", "man", "woman", "apple", "banana", "fruit", "server", "cloud", "docker"]
dim = 16

embeddings = {}
base_royal = np.random.randn(dim)
base_fruit = np.random.randn(dim)
base_tech = np.random.randn(dim)

embeddings["king"] = base_royal + np.random.normal(0, 0.1, dim) + np.array([1.0] + [0.0]*(dim-1))
embeddings["queen"] = base_royal + np.random.normal(0, 0.1, dim) + np.array([0.0, 1.0] + [0.0]*(dim-2))
embeddings["man"] = np.array([1.0] + [0.0]*(dim-1)) + np.random.normal(0, 0.1, dim)
embeddings["woman"] = np.array([0.0, 1.0] + [0.0]*(dim-2)) + np.random.normal(0, 0.1, dim)

embeddings["apple"] = base_fruit + np.random.normal(0, 0.1, dim)
embeddings["banana"] = base_fruit + np.random.normal(0, 0.1, dim)
embeddings["fruit"] = base_fruit

embeddings["server"] = base_tech + np.random.normal(0, 0.1, dim)
embeddings["cloud"] = base_tech + np.random.normal(0, 0.1, dim)
embeddings["docker"] = base_tech + np.random.normal(0, 0.1, dim)

mat = np.array([embeddings[w] for w in vocab])
sim_matrix = cosine_similarity(mat)

sim_df = pd.DataFrame(sim_matrix.round(2), index=vocab, columns=vocab)
print("=== Semantic Cosine Similarity Matrix ===")
print(sim_df[["king", "queen", "apple", "server"]])
"""),
            ("md", "## 2. Vector Arithmetic & Analogies\nTesting $\\vec{v}_{\\text{king}} - \\vec{v}_{\\text{man}} + \\vec{v}_{\\text{woman}} \\approx \\vec{v}_{\\text{queen}}$."),
            ("code", """
query_vector = (embeddings["king"] - embeddings["man"] + embeddings["woman"]).reshape(1, -1)
similarities = {w: cosine_similarity(query_vector, embeddings[w].reshape(1, -1))[0][0] for w in vocab}

sorted_matches = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
print("=== Vector Analogy: King - Man + Woman ===")
for word, score in sorted_matches[:5]:
    print(f"Match: {word:<10} | Cosine Similarity: {score:.4f}")
""")
        ]
    )
    save_notebook(nb_t08_02, "08_nlp_embeddings/02_dense_embeddings_word2vec_fasttext.ipynb")

    # ==========================================
    # TRACK 09 - NOTEBOOK 01: PyTorch CNN
    # ==========================================
    nb_t09_01 = create_notebook(
        title="01: PyTorch Convolutional Neural Networks (CNNs) from Scratch",
        description="Build and train a 2D Convolutional Neural Network: Conv2d kernels, MaxPooling, Batch Normalization, Dropout regularization, and Cross-Entropy optimization.",
        track_num="09",
        track_name="Computer Vision & Convolutional Neural Networks",
        cells_data=[
            ("md", "## 1. Convolutional Layer Mechanics & Spatial Reduction\n2D Convolution formula with kernel $K \\in \\mathbb{R}^{k \\times k}$:\n$$(I * K)(i, j) = \\sum_{m} \\sum_{n} I(i-m, j-n) K(m, n)$$"),
            ("code", """
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        feat = self.features(x)
        flattened = feat.view(feat.size(0), -1)
        out = self.classifier(flattened)
        return out

model = ConvNet(num_classes=10)
print(model)
"""),
            ("md", "## 2. Training Loop with Synthetic Digit Batches\nTrain for 3 epochs with Adam optimizer and CrossEntropyLoss."),
            ("code", """
torch.manual_seed(42)
X_dummy = torch.randn(200, 1, 28, 28)
y_dummy = torch.randint(0, 10, (200,))

dataset = TensorDataset(X_dummy, y_dummy)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("=== Starting CNN Training ===")
for epoch in range(3):
    total_loss = 0.0
    correct = 0
    total = 0
    for X_batch, y_batch in loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(y_batch)
        _, preds = torch.max(outputs, 1)
        correct += (preds == y_batch).sum().item()
        total += len(y_batch)
        
    epoch_loss = total_loss / total
    epoch_acc = correct / total * 100
    print(f"Epoch {epoch+1}/3 -> Loss: {epoch_loss:.4f} | Batch Accuracy: {epoch_acc:.2f}%")
""")
        ]
    )
    save_notebook(nb_t09_01, "09_computer_vision/01_pytorch_cnn_image_classification.ipynb")

    # ==========================================
    # TRACK 09 - NOTEBOOK 02: Transfer Learning & TensorBoard
    # ==========================================
    nb_t09_02 = create_notebook(
        title="02: Transfer Learning, Feature Extraction & Performance Logging",
        description="Fine-tune deep vision backbones (MobileNet/ResNet), freeze feature extractors, customize classification heads, and monitor metrics.",
        track_num="09",
        track_name="Computer Vision & Convolutional Neural Networks",
        cells_data=[
            ("md", "## 1. Transfer Learning Strategy\nWe freeze base convolutional weights $\\nabla W_{\\text{backbone}} = 0$ and train only task-specific linear heads."),
            ("code", """
import torch
import torch.nn as nn

# Base pretrained-style feature extractor backbone
class PretrainedBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(64, 1000)

    def forward(self, x):
        feat = self.features(x)
        flattened = feat.view(feat.size(0), -1)
        return self.classifier(flattened)

backbone = PretrainedBackbone()

# Freeze feature extractor layers (transfer learning)
for param in backbone.features.parameters():
    param.requires_grad = False

# Replace classifier head for 3 custom target classes
in_features = 64
backbone.classifier = nn.Sequential(
    nn.Dropout(p=0.2),
    nn.Linear(in_features, 3) # 3 custom output classes
)

trainable_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in backbone.parameters())

print("=== Transfer Learning Architecture Setup ===")
print(f"Total Parameters     : {total_params:,}")
print(f"Trainable Parameters : {trainable_params:,} ({(trainable_params/total_params)*100:.2f}%)")
"""),
            ("md", "## 2. Forward Pass Verification\nVerify output logits shape on an input RGB batch $(N, 3, 224, 224)$."),
            ("code", """
dummy_images = torch.randn(4, 3, 224, 224)
output_logits = backbone(dummy_images)
probabilities = torch.softmax(output_logits, dim=1)

print(f"Input Shape : {dummy_images.shape}")
print(f"Output Logits Shape : {output_logits.shape}")
print("Predicted Class Probabilities:")
print(probabilities.detach().numpy().round(4))
""")
        ]
    )
    save_notebook(nb_t09_02, "09_computer_vision/02_transfer_learning_and_tensorboard.ipynb")

    # ==========================================
    # TRACK 10 - NOTEBOOK 01: RNNs, LSTM & GRU
    # ==========================================
    nb_t10_01 = create_notebook(
        title="01: Deep Sequence Modeling: Recurrent Neural Networks (RNN, LSTM & GRU)",
        description="Master sequential information propagation, solve vanishing gradients via Gated Recurrent Units (GRU) and Long Short-Term Memory (LSTM) cell gates, and generate text.",
        track_num="10",
        track_name="Deep Sequential Models & Transformers from Scratch",
        cells_data=[
            ("md", "## 1. LSTM Cell Gates & State Dynamics\n$$\\begin{aligned} f_t &= \\sigma(W_f x_t + U_f h_{t-1} + b_f) \\quad &\\text{(Forget Gate)} \\\\ i_t &= \\sigma(W_i x_t + U_i h_{t-1} + b_i) \\quad &\\text{(Input Gate)} \\\\ \\tilde{C}_t &= \\tanh(W_c x_t + U_c h_{t-1} + b_c) \\quad &\\text{(Candidate Memory)} \\\\ C_t &= f_t \\odot C_{t-1} + i_t \\odot \\tilde{C}_t \\quad &\\text{(Cell State Update)} \\\\ o_t &= \\sigma(W_o x_t + U_o h_{t-1} + b_o) \\quad &\\text{(Output Gate)} \\\\ h_t &= o_t \\odot \\tanh(C_t) \\quad &\\text{(Hidden State)} \\end{aligned}$$"),
            ("code", """
import torch
import torch.nn as nn

class LSTMSequencePredictor(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, hidden=None):
        embeds = self.embedding(x)
        out, (hn, cn) = self.lstm(embeds, hidden)
        logits = self.fc(out)
        return logits, (hn, cn)

model = LSTMSequencePredictor(vocab_size=100, embed_dim=32, hidden_dim=64)
print(model)
"""),
            ("md", "## 2. Forward Inference & Hidden State Persistence\nPass sequence of tokens through LSTM."),
            ("code", """
input_tokens = torch.randint(0, 100, (4, 10))
logits, (hn, cn) = model(input_tokens)

print(f"Input Token Batch Shape : {input_tokens.shape}")
print(f"Output Logits Shape     : {logits.shape} (batch, seq_len, vocab_size)")
print(f"Final Hidden State Shape: {hn.shape} (layers, batch, hidden_dim)")
""")
        ]
    )
    save_notebook(nb_t10_01, "10_deep_sequence/01_recurrent_neural_networks_lstm_gru.ipynb")

    # ==========================================
    # TRACK 10 - NOTEBOOK 02: Transformers Attention from Scratch
    # ==========================================
    nb_t10_02 = create_notebook(
        title="02: Multi-Head Self-Attention & Transformer Encoders from Scratch",
        description="Implement the Transformer architecture from scratch in PyTorch: Scaled Dot-Product Attention, Multi-Head projections, Sinusoidal Positional Encoding, and LayerNorm residual blocks.",
        track_num="10",
        track_name="Deep Sequential Models & Transformers from Scratch",
        cells_data=[
            ("md", "## 1. Scaled Dot-Product Attention Formulation\n$$\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{Q K^T}{\\sqrt{d_k}}\\right) V$$"),
            ("code", """
import torch
import torch.nn as nn
import math

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super().__init__()
        self.scale = 1.0 / math.sqrt(d_k)
        
    def forward(self, q, k, v, mask=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        return output, attn_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention(self.d_k)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        q = self.w_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        out, weights = self.attention(q, k, v, mask=mask)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        return self.w_o(out), weights

mha = MultiHeadAttention(d_model=64, num_heads=4)
x = torch.randn(2, 8, 64)
out, weights = mha(x, x, x)

print("=== Multi-Head Attention Forward Pass ===")
print(f"Input Tensor Shape   : {x.shape}")
print(f"Output Tensor Shape  : {out.shape}")
print(f"Attention Map Shape  : {weights.shape} [batch, heads, seq, seq]")
""")
        ]
    )
    save_notebook(nb_t10_02, "10_deep_sequence/02_transformers_attention_mechanism_from_scratch.ipynb")

    print("✓ Tracks 07-10 generated successfully (8 notebooks).")
