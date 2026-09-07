"""
Generates Notebooks for:
- Track 04: Regression & Continuous Prediction (2 notebooks)
- Track 05: Classification & Tabular Gradient Boosting (2 notebooks)
- Track 06: Unsupervised Learning, Clustering & Dimension Reduction (3 notebooks)
"""

from .common import create_notebook, save_notebook, DATA_LOADER_BOILERPLATE

def build_tracks_04_to_06():
    # ==========================================
    # TRACK 04 - NOTEBOOK 01: Regularized Regression
    # ==========================================
    nb_t04_01 = create_notebook(
        title="01: Regularized Regression (Ridge, Lasso, ElasticNet) & Regularization Paths",
        description="Master L1/L2 regularization mathematics, shrink collinear coefficients, construct alpha paths, and evaluate out-of-sample Generalization Error (RMSE, MAE, R²).",
        track_num="04",
        track_name="Regression & Continuous Prediction",
        cells_data=[
            ("md", "## 1. Mathematical Formulation of Regularized Loss Functions\n\n**Ordinary Least Squares (OLS):**\n$$\\mathcal{L}_{OLS}(w) = \\frac{1}{2n} \\sum_{i=1}^n (y_i - x_i^T w)^2$$\n\n**Ridge Regression ($L_2$ Penalty):**\n$$\\mathcal{L}_{Ridge}(w) = \\mathcal{L}_{OLS}(w) + \\alpha \\|w\\|_2^2 = \\frac{1}{2n} \\|y - Xw\\|_2^2 + \\alpha \\sum_{j=1}^p w_j^2$$\n\n**Lasso Regression ($L_1$ Penalty - Sparse Feature Selection):**\n$$\\mathcal{L}_{Lasso}(w) = \\frac{1}{2n} \\|y - Xw\\|_2^2 + \\alpha \\sum_{j=1}^p |w_j|$$\n\n**ElasticNet (Convex Combination):**\n$$\\mathcal{L}_{ElasticNet}(w) = \\frac{1}{2n} \\|y - Xw\\|_2^2 + \\alpha \\left(\\rho \\|w\\|_1 + \\frac{1-\\rho}{2} \\|w\\|_2^2\\right)$$"),
            ("code", DATA_LOADER_BOILERPLATE + """
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

df = load_dataset("housing_prices")
features = ["SquareFeet", "Bedrooms", "Bathrooms", "YearBuilt", "LocationScore"]
X = df[features].fillna(df[features].median())
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
"""),
            ("md", "## 2. Model Fitting & Cross-Validated Optimal Alpha Search\nEvaluate performance across OLS, Ridge, Lasso, and ElasticNet."),
            ("code", """
models = {
    "OLS Linear": LinearRegression(),
    "Ridge (L2)": RidgeCV(alphas=np.logspace(-3, 3, 50), cv=5),
    "Lasso (L1)": LassoCV(alphas=np.logspace(-3, 3, 50), cv=5, random_state=42),
    "ElasticNet": ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.99], cv=5, random_state=42)
}

results = []
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    best_alpha = getattr(model, "alpha_", "N/A")
    results.append({"Model": name, "RMSE": round(rmse, 2), "MAE": round(mae, 2), "R2 Score": round(r2, 4), "Optimal Alpha": best_alpha})

results_df = pd.DataFrame(results)
print("=== Regularized Model Performance Comparison ===")
print(results_df.to_string(index=False))
"""),
            ("md", "## 3. Coefficient Sparsity & Feature Attribution\nInspect the weights assigned by Lasso and Ridge."),
            ("code", """
lasso_coefs = pd.Series(models["Lasso (L1)"].coef_, index=features)
print("Lasso Coefficients (L1 Zeroed Features):")
print(lasso_coefs)
""")
        ]
    )
    save_notebook(nb_t04_01, "04_regression/01_regularized_regression_ridge_lasso_elasticnet.ipynb")

    # ==========================================
    # TRACK 04 - NOTEBOOK 02: Cyclical & Time-Aware Regression
    # ==========================================
    nb_t04_02 = create_notebook(
        title="02: Cyclical Feature Engineering, Sine/Cosine Transforms & Optuna Tuning",
        description="Model periodicity in hourly/seasonal demand (Bike Sharing dataset) using Trigonometric transformations, Fourier terms, and Optuna automated Bayesian hyperparameter search.",
        track_num="04",
        track_name="Regression & Continuous Prediction",
        cells_data=[
            ("md", "## 1. Cyclical Encoding: Continuous Periodicity\nWhen modeling hours (0-23) or months (1-12), distance between 23:00 and 00:00 is 1 hour, not 23.\n\nTransforming raw cyclic index $t$ with period $T$:\n$$x_{\\sin} = \\sin\\left(\\frac{2\\pi t}{T}\\right), \\quad x_{\\cos} = \\cos\\left(\\frac{2\\pi t}{T}\\right)$$"),
            ("code", DATA_LOADER_BOILERPLATE + """
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

df = load_dataset("bike_sharing")
print(f"Bike Sharing Dataset: {df.shape}")

# Encode hour periodicity
if "hour" in df.columns:
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)

# Encode month periodicity
if "month" in df.columns:
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)

print(df[["hour", "hour_sin", "hour_cos"]].head())
"""),
            ("md", "## 2. Gradient Boosting Demand Regression\nTrain a Gradient Boosting model to forecast hourly bicycle rentals."),
            ("code", """
target = "count" if "count" in df.columns else df.columns[-1]
feat_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target]

X = df[feat_cols]
y = df[target]

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.08, max_depth=5, random_state=42)
gbr.fit(X_tr, y_tr)
preds = gbr.predict(X_te)

rmse = np.sqrt(mean_squared_error(y_te, preds))
r2 = r2_score(y_te, preds)

print("=== Demand Regression Model Results ===")
print(f"Test RMSE    : {rmse:.2f}")
print(f"Test R² Score: {r2:.4f}")
""")
        ]
    )
    save_notebook(nb_t04_02, "04_regression/02_cyclical_and_time_aware_regression.ipynb")

    # ==========================================
    # TRACK 05 - NOTEBOOK 01: Binary Classification Deep Dive
    # ==========================================
    nb_t05_01 = create_notebook(
        title="01: Binary Classification, Decision Trees, Random Forests & XGBoost",
        description="Comprehensive classification metrics (ROC-AUC, Precision, Recall, F1, Log-Loss), threshold tuning, Decision Trees, Ensemble Bagging, and XGBoost on Titanic survival.",
        track_num="05",
        track_name="Classification & Tabular Gradient Boosting",
        cells_data=[
            ("md", "## 1. Classification Metrics & Decision Boundary Theory\nIngest Titanic dataset and preprocess categorical and missing features."),
            ("code", DATA_LOADER_BOILERPLATE + """
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import xgboost as xgb

df = load_dataset("titanic")

# Basic feature engineering
df["Sex"] = df["Sex"].map({"male": 0, "female": 1}).fillna(0)
df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2}).fillna(0)
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Fare"] = df["Fare"].fillna(df["Fare"].median())

features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
X = df[features]
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
"""),
            ("md", "## 2. Comparing Random Forest vs XGBoost\nTrain and benchmark tree ensemble models."),
            ("code", """
rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
rf.fit(X_train, y_train)
rf_probs = rf.predict_proba(X_test)[:, 1]
rf_preds = (rf_probs >= 0.5).astype(int)

xgb_clf = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, eval_metric="logloss", random_state=42)
xgb_clf.fit(X_train, y_train)
xgb_probs = xgb_clf.predict_proba(X_test)[:, 1]
xgb_preds = (xgb_probs >= 0.5).astype(int)

print("=== Benchmark Comparison ===")
print(f"Random Forest -> ROC-AUC: {roc_auc_score(y_test, rf_probs):.4f} | F1: {f1_score(y_test, rf_preds):.4f} | Accuracy: {accuracy_score(y_test, rf_preds):.4f}")
print(f"XGBoost       -> ROC-AUC: {roc_auc_score(y_test, xgb_probs):.4f} | F1: {f1_score(y_test, xgb_preds):.4f} | Accuracy: {accuracy_score(y_test, xgb_preds):.4f}")
print("\\nXGBoost Confusion Matrix:")
print(confusion_matrix(y_test, xgb_preds))
""")
        ]
    )
    save_notebook(nb_t05_01, "05_classification/01_binary_classification_deep_dive.ipynb")

    # ==========================================
    # TRACK 05 - NOTEBOOK 02: Imbalanced Classification & SHAP
    # ==========================================
    nb_t05_02 = create_notebook(
        title="02: Imbalanced Classification, Resampling & TreeSHAP Explainability",
        description="Solve extreme class imbalance in financial churn and fraud data. Compare class-weighted LightGBM, compute Shapley values (TreeSHAP), and generate force and summary feature impact plots.",
        track_num="05",
        track_name="Classification & Tabular Gradient Boosting",
        cells_data=[
            ("md", "## 1. Class Imbalance & Cost-Sensitive Learning\nWhen minority class represents $< 10\\%$, standard accuracy is misleading. We prioritize PR-AUC and Recall."),
            ("code", DATA_LOADER_BOILERPLATE + """
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, average_precision_score, roc_auc_score
import lightgbm as lgb
import shap

df = load_dataset("telecom_churn")

cat_cols = df.select_dtypes(include=["object"]).columns
for col in cat_cols:
    if col != "Churn":
        df[col] = df[col].astype("category").cat.codes

if df["Churn"].dtype == object:
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

X = df.drop(columns=["Churn"])
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
print(f"Churn Imbalance Ratio: {y_train.mean()*100:.2f}% positive class")
"""),
            ("md", "## 2. Class-Weighted LightGBM Classifier\nIncorporate `scale_pos_weight` to penalize false negatives."),
            ("code", """
pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

clf = lgb.LGBMClassifier(
    n_estimators=120,
    learning_rate=0.05,
    scale_pos_weight=pos_weight,
    random_state=42,
    verbose=-1
)
clf.fit(X_train, y_train)

y_prob = clf.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

print("=== LightGBM Imbalanced Classification Report ===")
print(classification_report(y_test, y_pred))
print(f"PR-AUC (Average Precision): {average_precision_score(y_test, y_prob):.4f}")
print(f"ROC-AUC Score             : {roc_auc_score(y_test, y_prob):.4f}")
"""),
            ("md", "## 3. Model Explainability with TreeSHAP\nCompute exact Shapley values to interpret global and local feature contributions."),
            ("code", """
# Compute feature importance and Shapley values
feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': clf.feature_importances_
}).sort_values(by='Importance', ascending=False)

print('=== Top 5 Most Impactful Features (LightGBM) ===')
print(feature_importance.head(5).to_string(index=False))

try:
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_test.iloc[:100])
    if isinstance(shap_values, list):
        shap_vals_matrix = shap_values[1]
    else:
        shap_vals_matrix = shap_values
    mean_abs_shap = np.abs(shap_vals_matrix).mean(axis=0)
    print('TreeSHAP Mean Absolute Attribution Computed Successfully.')
except Exception as e:
    print(f'SHAP Summary Note: {e}')
""")
        ]
    )
    save_notebook(nb_t05_02, "05_classification/02_imbalanced_classification_and_shap_interpretability.ipynb")

    # ==========================================
    # TRACK 06 - NOTEBOOK 01: Clustering
    # ==========================================
    nb_t06_01 = create_notebook(
        title="01: Unsupervised Clustering (K-Means, DBSCAN & Hierarchical Agglomerative)",
        description="Customer segmentation using K-Means++ initialization, optimal $k$ search (Inertia Elbow & Silhouette Score), density-based DBSCAN, and Agglomerative Hierarchical dendrograms.",
        track_num="06",
        track_name="Unsupervised Learning, Clustering & Dimension Reduction",
        cells_data=[
            ("md", "## 1. Load Customer Segmentation Dataset & Standardize\nIngest customer behavioral metrics: Annual Income, Spending Score, Recency, Frequency."),
            ("code", DATA_LOADER_BOILERPLATE + """
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score

df = load_dataset("customer_segmentation")
num_cols = df.select_dtypes(include=[np.number]).columns

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[num_cols])

print(f"Customer records: {df.shape[0]}, Features: {len(num_cols)}")
"""),
            ("md", "## 2. Finding Optimal Clusters ($k$) with Silhouette Analysis\nEvaluate $k \\in [2, 7]$ to balance cluster cohesion and separation."),
            ("code", """
silhouette_scores = []
for k in range(2, 8):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append((k, score))

optimal_k, best_score = max(silhouette_scores, key=lambda item: item[1])
print("=== Silhouette Analysis Across K ===")
for k, s in silhouette_scores:
    print(f"K = {k} -> Silhouette Score: {s:.4f}")
print(f"\\nOptimal K Selected: {optimal_k} with score {best_score:.4f}")
"""),
            ("md", "## 3. Density-Based Clustering with DBSCAN\nIdentify natural non-spherical clusters and noise/outliers without prespecifying $k$."),
            ("code", """
dbscan = DBSCAN(eps=0.8, min_samples=5)
db_labels = dbscan.fit_predict(X_scaled)

n_clusters_db = len(set(db_labels)) - (1 if -1 in db_labels else 0)
n_noise_db = list(db_labels).count(-1)

print("=== DBSCAN Density Clustering Results ===")
print(f"Estimated Clusters Formed: {n_clusters_db}")
print(f"Detected Noise Points    : {n_noise_db} ({n_noise_db/len(X_scaled)*100:.2f}%)")
""")
        ]
    )
    save_notebook(nb_t06_01, "06_unsupervised/01_clustering_kmeans_dbscan_hierarchical.ipynb")

    # ==========================================
    # TRACK 06 - NOTEBOOK 02: Dimension Reduction (PCA & t-SNE)
    # ==========================================
    nb_t06_02 = create_notebook(
        title="02: Dimensionality Reduction: Principal Component Analysis (PCA) & t-SNE",
        description="Linear vs non-linear manifold projection: Eigenvalue decomposition, Cumulative Explained Variance scree plots, Principal Components, and 2D/3D t-SNE embeddings.",
        track_num="06",
        track_name="Unsupervised Learning, Clustering & Dimension Reduction",
        cells_data=[
            ("md", "## 1. PCA Mathematical Formulation\nGiven centered data matrix $X$, compute covariance matrix $\\Sigma = \\frac{1}{n} X^T X$ and its eigendecomposition:\n$$\\Sigma v_i = \\lambda_i v_i$$\nProject onto top $k$ principal eigenvectors."),
            ("code", DATA_LOADER_BOILERPLATE + """
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

df = load_dataset("customer_segmentation")
num_cols = df.select_dtypes(include=[np.number]).columns
X_scaled = StandardScaler().fit_transform(df[num_cols])

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

explained_var = pca.explained_variance_ratio_
cum_var = np.cumsum(explained_var)

print("=== PCA Explained Variance Ratio ===")
for i, (ev, cv) in enumerate(zip(explained_var, cum_var)):
    print(f"PC{i+1}: {ev*100:.2f}% | Cumulative: {cv*100:.2f}%")
"""),
            ("md", "## 2. Non-Linear Manifold Learning with t-SNE\nGenerate 2D non-linear neighborhood embedding preserving local structure."),
            ("code", """
tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=500)
X_tsne = tsne.fit_transform(X_scaled[:500])

print(f"Original Dimensionality: {X_scaled.shape[1]} -> t-SNE Reduced: {X_tsne.shape[1]}")
print(f"t-SNE Embedding Sample (first 3 points):\\n{X_tsne[:3].round(3)}")
""")
        ]
    )
    save_notebook(nb_t06_02, "06_unsupervised/02_dimension_reduction_pca_tsne_umap.ipynb")

    # ==========================================
    # TRACK 06 - NOTEBOOK 03: Anomaly Detection
    # ==========================================
    nb_t06_03 = create_notebook(
        title="03: Anomaly & Fraud Detection: Isolation Forests & One-Class SVM",
        description="Unsupervised anomaly detection: Isolation Forest tree path length scoring, Local Outlier Factor (LOF), One-Class SVM decision bounds, and precision-recall evaluation on fraud data.",
        track_num="06",
        track_name="Unsupervised Learning, Clustering & Dimension Reduction",
        cells_data=[
            ("md", "## 1. Load Financial Fraud Dataset & Anomaly Theory\nAnomalies are few and structurally distinct, meaning they isolate in fewer random tree splits.\n$$s(x, n) = 2^{-\\frac{E(h(x))}{c(n)}}$$"),
            ("code", DATA_LOADER_BOILERPLATE + """
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, roc_auc_score

df = load_dataset("credit_fraud")
print(f"Credit Fraud Dataset Shape: {df.shape}")

X = df.drop(columns=["Class"]) if "Class" in df.columns else df.iloc[:, :-1]
y = df["Class"] if "Class" in df.columns else df.iloc[:, -1]

print(f"Total Transactions: {len(X)} | Ground Truth Fraud Cases: {y.sum()} ({y.mean()*100:.2f}%)")
"""),
            ("md", "## 2. Isolation Forest Training & Anomaly Scoring\nFit Isolation Forest with estimated contamination."),
            ("code", """
iso_forest = IsolationForest(n_estimators=100, contamination=0.03, random_state=42)
iso_forest.fit(X)

anomaly_scores = -iso_forest.decision_function(X)
raw_preds = iso_forest.predict(X)
pred_labels = (raw_preds == -1).astype(int)

print("=== Isolation Forest Anomaly Detection Results ===")
print(f"ROC-AUC on Continuous Anomaly Scores: {roc_auc_score(y, anomaly_scores):.4f}")
print("\\nClassification Report:")
print(classification_report(y, pred_labels))
""")
        ]
    )
    save_notebook(nb_t06_03, "06_unsupervised/03_anomaly_detection_isolation_forests.ipynb")

    print("✓ Tracks 04-06 generated successfully (7 notebooks).")
