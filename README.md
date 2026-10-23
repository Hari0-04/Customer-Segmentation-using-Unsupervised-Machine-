# 🧩 Customer Segmentation using Unsupervised Machine Learning

An end-to-end customer segmentation project. It cleans a customer dataset (2,200+ records),
imputes missing values, scales features, reduces dimensionality with PCA (95% variance),
benchmarks four clustering algorithms, profiles the resulting segments, and presents everything
in an interactive Streamlit dashboard.

---

## 1. Project Overview

| Stage | What happens |
|---|---|
| **Data cleaning** | Median imputation for numeric columns, mode imputation for categorical columns |
| **EDA** | Missing-value report, distributions, correlation matrix, category counts |
| **Preprocessing** | One-hot encoding of categoricals, standardisation of numeric features |
| **Dimensionality reduction** | PCA retaining 95% of variance (21 encoded features → 13 components) |
| **Modelling** | K-Means, Agglomerative (Ward), DBSCAN, Gaussian Mixture |
| **Model selection** | Elbow Method, Silhouette Score, Davies-Bouldin, Calinski-Harabasz, GMM BIC |
| **Profiling** | Segment size, feature means, demographics, z-score index, radar chart |
| **Delivery** | Streamlit app with visualisation, segment comparison, prediction and CSV export |

---

## 2. Project Structure

```
customer_segmentation/
├── app.py              # Streamlit dashboard (7 tabs)
├── segmentation.py     # Core ML pipeline (preprocess, PCA, cluster, benchmark, profile)
├── generate_data.py    # Synthetic dataset generator (2,200 rows, ~5% missing values)
├── train.py            # Headless pipeline run; saves results to outputs/
├── requirements.txt    # Python dependencies
├── README.md           # This file
├── data/
│   └── customers.csv   # Sample dataset (auto-generated if missing)
└── outputs/            # Created by train.py (segmented customers, profiles, benchmark)
```

---

## 3. Requirements

- Python **3.9 or newer**
- Packages: `streamlit>=1.50`, `pandas`, `numpy`, `scikit-learn>=1.2`, `plotly`

---

## 4. Setup and Run

### Step 1 – Open a terminal in the project folder
```bash
cd customer_segmentation
```

### Step 2 – Create and activate a virtual environment
```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows (PowerShell)
venv\Scripts\Activate.ps1

# Windows (cmd)
venv\Scripts\activate.bat
```

### Step 3 – Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 – (Optional) Regenerate the sample data
```bash
python generate_data.py
```

### Step 5 – (Optional) Run the pipeline from the command line
```bash
python train.py
```
Prints the benchmark table and segment profiles, and saves:
- `outputs/customers_segmented.csv`
- `outputs/segment_profiles.csv`
- `outputs/model_benchmark.csv`

### Step 6 – Launch the dashboard
```bash
streamlit run app.py
```
Open **http://localhost:8501** if the browser does not open automatically.
Stop the app with `Ctrl + C`.

---

## 5. Dataset

`data/customers.csv` is synthetic but behaves like real retail data. It contains five hidden
behavioural groups that the models are expected to rediscover.

| Column | Type | Description |
|---|---|---|
| `CustomerID` | ID | Unique identifier (excluded from modelling) |
| `Age` | numeric | Customer age |
| `Gender`, `Marital_Status`, `Education`, `City_Tier` | categorical | Demographics |
| `Annual_Income_k` | numeric | Annual income (thousands) |
| `Spending_Score` | numeric | 1–100 spending score |
| `Tenure_Months` | numeric | Months as a customer |
| `Purchase_Frequency` | numeric | Purchases per year |
| `Avg_Order_Value` | numeric | Average basket value |
| `Online_Purchase_Ratio` | numeric | Share of purchases made online (0–1) |
| `Recency_Days` | numeric | Days since last purchase |
| `Discount_Usage_Pct` | numeric | Share of purchases using a discount |

About 975 cells are deliberately missing so that imputation is exercised.

**Using your own data:** choose *Upload CSV* in the sidebar. Include numeric and/or categorical
feature columns. A `CustomerID` column is optional and is ignored if present.

---

## 6. Methodology

1. **Imputation** – `SimpleImputer` with median (numeric) and most-frequent (categorical).
2. **Encoding and scaling** – `OneHotEncoder` for categoricals, `StandardScaler` for numerics.
3. **PCA** – `PCA(n_components=0.95)` keeps the smallest number of components explaining 95% of variance.
4. **Clustering** – each algorithm is tuned:
   - K-Means, Agglomerative, GMM: k = 2 … *max k*, choosing the k with the best silhouette.
   - DBSCAN: grid over `eps` (k-distance percentiles) and `min_samples` (5, 10, 15, 20); only
     configurations with 2–10 clusters and ≤ 30% noise are considered.
5. **Elbow Method** – K-Means inertia curve with automatic knee detection (furthest point from the
   line joining the first and last points).
6. **Ranking** – by **Adjusted Silhouette = Silhouette × (1 − noise share)**. DBSCAN's raw silhouette
   ignores noise points, so this stops it winning by discarding customers.
7. **Profiling** – per-segment feature means, top demographic categories, and a z-score index versus the
   overall population. Segments are named automatically from their two most distinctive features.

---

## 7. Dashboard Tabs

| Tab | Contents |
|---|---|
| 📊 Data & EDA | Dataset preview, missing values, correlation heatmap, distributions |
| 📉 PCA | Explained-variance chart and component loadings |
| 🧪 Model Benchmark | Ranked model table, elbow plot, silhouette vs k, GMM BIC, DBSCAN grid |
| 🎯 Cluster Visualization | PCA 2D, PCA 3D, or any feature pair, coloured by segment |
| 👥 Segment Profiles | Size chart, z-score heatmap, profile table, radar chart, recommendations, CSV download |
| ⚖️ Compare Segments | Pick any two segments for side-by-side z-scores and % differences |
| 🔮 Predict Segment | Enter a new customer's details and get the nearest segment |

**Sidebar controls:** data source, PCA variance (80–99%), max clusters to evaluate, clustering model
(or *Auto*), and a manual cluster-count override.

---

## 8. Sample Results

| Model | Clusters | Noise % | Silhouette | Adj. Silhouette |
|---|---|---|---|---|
| **K-Means** | 5 | 0 | 0.387 | **0.387** |
| Gaussian Mixture | 5 | 0 | 0.385 | 0.385 |
| Agglomerative | 5 | 0 | 0.382 | 0.382 |
| DBSCAN | 3 | 27.8 | 0.420 | 0.303 |

The Elbow Method also suggests k = 5. The recovered segments are:

1. High-discount, lower-spending families
2. Young, online-first, high-frequency shoppers
3. Affluent, high-order-value loyalists
4. Older, long-tenure customers
5. Dormant, high-recency (at-risk) customers

---

## 9. Troubleshooting

| Problem | Fix |
|---|---|
| `streamlit: command not found` | Activate the virtual environment, or run `python -m streamlit run app.py` |
| `use_container_width` deprecation warnings | Upgrade Streamlit: `pip install --upgrade streamlit` |
| `sparse_output` error from `OneHotEncoder` | Upgrade scikit-learn: `pip install --upgrade scikit-learn` |
| Port 8501 already in use | `streamlit run app.py --server.port 8502` |
| Slow first load | Results are cached; later interactions are fast |

---

## 10. Possible Extensions

- Add RFM (Recency, Frequency, Monetary) feature engineering
- Try UMAP or t-SNE for visualisation
- Add stability checks (bootstrap re-clustering, adjusted Rand index)
- Save the fitted pipeline with `joblib` and serve predictions via an API
- Connect to a real data source (SQL, data warehouse)

---

## 11. Tech Stack

Python · pandas · NumPy · scikit-learn · Plotly · Streamlit
# Customer-Segmentation-using-Unsupervised-Machine-
