# Self-Healing ML Pipeline: Complete Technical Documentation

## Table of Contents

1. [Project Overview and Motivation](#1-project-overview-and-motivation)
2. [Architecture Deep Dive](#2-architecture-deep-dive)
3. [Data Layer — Generation and Preprocessing](#3-data-layer--generation-and-preprocessing)
4. [Feature Store — Feast Integration and Point-in-Time Correctness](#4-feature-store--feast-integration-and-point-in-time-correctness)
5. [Training Pipeline — Multi-Architecture Sweeps and MLflow](#5-training-pipeline--multi-architecture-sweeps-and-mlflow)
6. [Model Evaluation and Promotion Logic](#6-model-evaluation-and-promotion-logic)
7. [Drift Detection — Population Stability Index (PSI)](#7-drift-detection--population-stability-index-psi)
8. [Serving Layer — FastAPI and Real-Time Inference](#8-serving-layer--fastapi-and-real-time-inference)
9. [Monitoring — Prometheus and Grafana](#9-monitoring--prometheus-and-grafana)
10. [The Self-Healing Loop — How It All Connects](#10-the-self-healing-loop--how-it-all-connects)
11. [Configuration Reference](#11-configuration-reference)
12. [Scaling Up — From Laptop to Production](#12-scaling-up--from-laptop-to-production)

---

## 1. Project Overview and Motivation

### What Problem Does This Solve?

Machine learning models degrade in production. This is not a possibility — it is a certainty. The data distributions that a model was trained on will shift over time due to seasonality, changing user behavior, market conditions, adversarial adaptation, and dozens of other factors. When this happens, the model's predictions become unreliable, but the model itself has no mechanism to flag this. It will keep serving predictions with the same confidence scores even as its actual accuracy declines.

In traditional ML operations, a human data scientist monitors dashboards, notices degradation, investigates the root cause, retrains a model, validates it, and deploys it. This cycle takes days to weeks. During that window, the system is serving increasingly wrong predictions.

A self-healing pipeline eliminates that gap. It continuously monitors its own prediction quality and the statistical properties of incoming data. When it detects that the world has changed (data drift), it autonomously retrains on recent data, validates the new model against rigorous quality gates, and promotes it to production — all without human intervention.

### Why Fraud Detection?

Credit card fraud detection is the canonical use case for self-healing pipelines because:

1. **Adversarial drift is guaranteed.** Fraudsters actively adapt their strategies to evade detection. A model trained on January's fraud patterns will miss March's novel attack vectors.

2. **The cost of stale models is measured in money.** Every hour a degraded fraud model runs, real financial losses accumulate — either from missed fraud (false negatives) or from blocking legitimate transactions (false positives) that drives customers away.

3. **Class imbalance makes drift detection critical.** With a ~3.5% fraud rate, even small distribution shifts in the majority class can dramatically change the model's decision boundary.

4. **Precision-recall tradeoffs are non-trivial.** Unlike many classification tasks where accuracy is sufficient, fraud detection requires careful balancing — too many false positives and the operations team stops trusting the system; too many false negatives and money walks out the door.

### Technology Stack Rationale

| Component | Technology | Why This Choice |
|-----------|-----------|-----------------|
| Data Generation | NumPy + Pandas | Full control over distribution parameters and drift simulation. No dependency on external datasets. |
| Feature Store | Feast | Industry standard for point-in-time correct feature retrieval. Prevents training-serving skew, which is one of the most insidious bugs in production ML. |
| Experiment Tracking | MLflow | De facto standard for ML experiment tracking. Logs parameters, metrics, artifacts, and models in a queryable format. |
| Model Training | scikit-learn, LightGBM, XGBoost | Covers the spectrum from interpretable (logistic regression) to high-performance (gradient boosted trees). All support the scikit-learn API, enabling uniform training code. |
| Drift Detection | Custom PSI Implementation | PSI (Population Stability Index) is the standard statistical test for distribution shift in production ML. Custom implementation gives full control over binning strategy and threshold tuning. |
| Serving | FastAPI | Async-capable, Pydantic validation built-in, automatic OpenAPI docs, native Prometheus integration. The standard for Python ML serving. |
| Monitoring | Prometheus + Grafana | The industry standard for time-series metrics and dashboarding. Prometheus pull-based scraping, Grafana for visualization and alerting. |
| Orchestration | Custom Python | Keeps the self-healing logic transparent and debuggable. No heavy framework overhead (Airflow/Kubeflow are overkill for the core loop). |
| Infrastructure | Docker Compose | Reproducible multi-service deployment. One command to spin up the entire stack. |

---

## 2. Architecture Deep Dive

### System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SELF-HEALING LOOP                            │
│                                                                     │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────┐  │
│  │  Data     │───>│  Feature  │───>│ Training │───>│  Evaluation  │  │
│  │ Generator │    │  Store    │    │ Pipeline │    │  & Promotion │  │
│  └──────────┘    │ (Feast)   │    │ (MLflow) │    │              │  │
│                  └──────────┘    └──────────┘    └──────┬───────┘  │
│                                                         │          │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐          │          │
│  │  Drift   │<───│Production │<───│ Serving  │<─────────┘          │
│  │ Detector │    │  Data     │    │ (FastAPI)│                     │
│  │  (PSI)   │    │  Stream   │    └──────────┘                     │
│  └────┬─────┘    └──────────┘                                      │
│       │                                                             │
│       │ drift detected?                                             │
│       │ ──── yes ──> trigger retrain (loops back to Training)       │
│       │ ──── no  ──> sleep, check again                             │
│       │                                                             │
│  ┌────┴─────┐    ┌──────────┐                                      │
│  │Prometheus │───>│ Grafana  │                                      │
│  │ Metrics   │    │Dashboard │                                      │
│  └──────────┘    └──────────┘                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Data Flow

The pipeline operates in two distinct phases:

**Phase 1: Bootstrap (cold start)**
```
generate_training_data() → validate_schema() → feast.initialize()
→ feast.get_historical_features() → split_data()
→ run_hyperparameter_sweep() → evaluate_model() → attempt_promotion()
→ model saved to registry, promoted to active
```

**Phase 2: Continuous Monitoring (self-healing loop)**
```
while running:
    new_data = generate_drifted_data()  # or receive from production stream
    drift_report = drift_detector.check(new_data)
    
    if drift_report.should_retrain:
        combined_data = old_training_data + new_data
        retrain → evaluate → promote_if_better
        update_drift_detector_reference()
    
    push_metrics_to_prometheus()
    sleep(interval)
```

### Module Dependency Graph

```
config/settings.py          ← imported by everything
    │
src/data/generator.py       ← standalone, no internal deps
src/data/preprocessor.py    ← depends on settings
    │
src/features/store.py       ← depends on settings, feast
    │
src/training/trainer.py     ← depends on settings, mlflow, sklearn
src/training/evaluator.py   ← depends on settings, sklearn
    │
src/drift/detector.py       ← depends on settings, numpy
    │
src/monitoring/metrics.py   ← depends on prometheus_client
    │
src/serving/model_loader.py ← depends on settings, joblib
src/serving/app.py          ← depends on model_loader, metrics, settings
    │
src/pipeline/promoter.py    ← depends on model_loader, evaluator, metrics
src/pipeline/orchestrator.py ← depends on everything above
    │
run_pipeline.py             ← entry point, depends on orchestrator + serving
```

---

## 3. Data Layer — Generation and Preprocessing

### Why Synthetic Data?

We generate synthetic data rather than using a static dataset for three reasons:

1. **Controllable drift.** We can precisely dial how much distribution shift to inject, which lets us test the self-healing pipeline under controlled conditions. With a static dataset, you would need to manually create train/test splits that simulate drift — which is fragile and hard to parameterize.

2. **Reproducibility.** Seeded random generators mean every run with the same seed produces identical data. This makes debugging deterministic.

3. **No data licensing issues.** Real fraud datasets (like the Kaggle credit card dataset) have licensing constraints and are also relatively small. Synthetic generation lets us scale to any size.

### How the Generator Works

#### File: `src/data/generator.py`

The generator constructs transactions in four stages:

**Stage 1: Base Feature Generation (`_base_transaction_features`)**

Each feature is drawn from a distribution chosen to mimic real transaction patterns:

```python
tx_amount = np.exp(rng.normal(3.8, 1.1, n))  # log-normal spend
```

Transaction amounts follow a log-normal distribution because real spending data is heavily right-skewed — most transactions are small (coffee, groceries) with a long tail of large purchases. The parameters (mu=3.8, sigma=1.1) produce a median around $45 and a 99th percentile around $3,000, which is realistic.

```python
hours = rng.normal(14, 4, n)  # peak around 2pm
```

Transaction hours follow a Gaussian centered at 2 PM because that is when transaction volume peaks in real payment networks. The standard deviation of 4 hours means most transactions fall between 6 AM and 10 PM, with a thin tail into late night.

```python
distance_from_home = np.abs(rng.exponential(8.0, n))
```

Distance from home follows an exponential distribution because most transactions happen near the cardholder's home (grocery store, gas station), with exponentially decreasing frequency at greater distances. The scale parameter of 8.0 means the average distance is 8 km, but 95% of transactions are within ~24 km.

The engineered features (`tx_frequency_1h`, `tx_amount_avg_7d`, `tx_amount_std_7d`) simulate what you would compute from a real transaction history using window aggregations. In production, these would come from the feature store; here we approximate them from the transaction-level data.

**Stage 2: Fraud Label Assignment (`_assign_fraud_labels`)**

Rather than randomly assigning fraud labels (which would produce a dataset where fraud is independent of features — useless for training), we compute a fraud probability score based on known risk indicators:

```python
score += np.where(df["ratio_to_median_price"] > 2.0, 0.3, 0.0)  # unusual amount
score += np.where((df["is_online"] == 1) & (df["is_pin_used"] == 0), 0.2, 0.0)  # risky channel
score += np.where((df["tx_hour"] < 5) | (df["tx_hour"] > 22), 0.15, 0.0)  # unusual time
```

The individual score contributions are then passed through a sigmoid function and calibrated to hit the target fraud ratio (~3.5%):

```python
probs = 1 / (1 + np.exp(-(score - 0.3) * 5))
probs = probs * (fraud_ratio / probs.mean())
```

This ensures that fraud labels correlate with the features a model would learn from, creating a realistic signal-to-noise ratio. The calibration step adjusts the overall rate — without it, the sigmoid would produce whatever fraud rate the raw scores happen to imply.

**Stage 3: Fraud Pattern Amplification (`_inject_fraud_patterns`)**

After labels are assigned, we amplify the fraudulent transactions to make them more distinct:

```python
df.loc[fraud_mask, "tx_amount"] *= rng.uniform(1.5, 4.0, n_fraud)
df.loc[fraud_mask, "is_chip_used"] = rng.binomial(1, 0.25, n_fraud)  # less chip usage
df.loc[fraud_mask, "is_pin_used"] = rng.binomial(1, 0.10, n_fraud)   # almost never PIN
```

This creates a two-way dependency: fraud labels depend on features (from Stage 2), and features are then modified conditioned on labels (Stage 3). This mirrors reality where fraudulent transactions genuinely look different from legitimate ones — larger amounts, no PIN, unusual locations.

**Stage 4: Timestamp and Entity Assignment (`_add_timestamps`)**

```python
offsets = np.sort(rng.uniform(0, 30 * 24 * 3600, n))  # ~30 days
timestamps = [start_date + timedelta(seconds=float(s)) for s in offsets]
df["entity_id"] = [f"user_{i % 800:04d}" for i in range(n)]
```

Timestamps are uniformly distributed across a 30-day window and then sorted (transactions arrive in chronological order). Entity IDs cycle through ~800 users, simulating a realistic cardinality where each user has multiple transactions.

### Drift Simulation (`generate_drifted_data`)

The `drift_intensity` parameter controls how much the distributions shift:

```python
amount_shift = 1.0 + 0.4 * drift_intensity
df["tx_amount"] *= amount_shift
```

At `drift_intensity=1.0`:
- Transaction amounts increase by 40% (inflation, higher-value targets)
- 15% of transactions shift to online retail (changing merchant mix)
- Transaction hours shift later by 2 hours (behavioral change)
- Distance from home increases by 30% (geographic expansion)
- Transaction frequency increases (velocity change)
- Fraud ratio increases by 50% (adversarial escalation)

At `drift_intensity=2.0`, all shifts double. The orchestrator increases drift intensity with each retraining cycle, simulating an increasingly adversarial environment that forces the pipeline to keep adapting.

### Preprocessing

#### File: `src/data/preprocessor.py`

**Schema Validation (`validate_schema`)**

Before any data enters the training pipeline, we verify:
1. All required columns exist (both features and target)
2. Feature columns are coerced to numeric types
3. Any NaN values introduced by coercion are filled with column medians
4. The target column is cast to integer

This is a defensive boundary — the training code assumes clean data, and this function enforces that contract.

**Stratified Splitting (`split_data`)**

```python
train_df, test_df = train_test_split(
    df, test_size=test_ratio, random_state=seed, stratify=df[TARGET_COLUMN]
)
```

The `stratify` parameter is critical with imbalanced classes. Without it, a random 80/20 split could produce a test set with 0% fraud or 8% fraud purely by chance, leading to unreliable evaluation metrics. Stratification guarantees the fraud ratio is preserved in both splits.

**Feature Statistics (`compute_feature_stats`)**

Computes per-feature summary statistics (mean, std, min, max, quartiles) and the target rate. These serve two purposes:
1. Baseline for drift detection — we compare production data stats against training data stats
2. Monitoring — pushed to Prometheus gauges for dashboard visualization

---

## 4. Feature Store — Feast Integration and Point-in-Time Correctness

### Why a Feature Store Matters

#### File: `src/features/store.py`

The feature store addresses one of the most subtle and dangerous bugs in production ML: **training-serving skew caused by temporal leakage**.

Consider this scenario without a feature store:

1. You compute `tx_amount_avg_7d` (average transaction amount over the past 7 days) for each transaction in your training data.
2. You compute it using a simple groupby + rolling mean on the full dataset.
3. For a transaction on January 15, this rolling mean might inadvertently include transactions from January 16–22 (future data) because you computed the aggregate on the entire dataframe rather than respecting the temporal boundary.

The model learns to rely on this "future information" during training, but at serving time, the feature is computed correctly (only using past data). The result: the model performs brilliantly in offline evaluation but poorly in production, and the discrepancy is extremely hard to diagnose.

**Point-in-time joins** solve this by ensuring that for each entity (user) at each timestamp, the feature retrieval only returns data that was available *at or before that timestamp*.

### How Feast Is Configured

The `FeatureStoreManager` class dynamically generates the Feast repository:

```python
def _ensure_repo(self, parquet_path: str):
    # Write feature_store.yaml (provider, online store config)
    # Write features.py (entity definitions, feature views, data sources)
    self._store = FeatureStore(repo_path=str(self.repo_path))
```

**Entity Definition:**
```python
transaction_entity = Entity(
    name="entity_id",
    description="User/account identifier",
)
```

The entity is the primary key for feature lookups. In our case, it is the user/account ID. Each user has multiple transactions over time.

**Feature View:**
```python
transaction_features = FeatureView(
    name="transaction_features",
    entities=[transaction_entity],
    ttl=timedelta(days=90),
    schema=[...],
    source=tx_source,
    online=True,
)
```

The `ttl` (time-to-live) of 90 days means features older than 90 days are considered stale and will not be returned. This prevents the model from seeing ancient data that may no longer be relevant.

The `online=True` flag enables materialization to the online store (SQLite locally) for low-latency serving.

### Historical vs. Online Feature Retrieval

**Historical (for training):**
```python
job = self._store.get_historical_features(
    entity_df=entity_df[["entity_id", "event_timestamp"]],
    features=feature_refs,
)
```

This performs the point-in-time join. For each (entity_id, event_timestamp) pair, Feast finds the most recent feature values that existed *before* that timestamp.

**Online (for serving):**
```python
result = self._store.get_online_features(
    features=feature_refs,
    entity_rows=entity_rows,
)
```

Online retrieval returns the *latest* feature values for each entity. This is used at inference time when you need features for a new transaction in real-time. The materialization step (`materialize()`) pushes the latest offline features into the online store.

### Materialization

```python
def materialize(self, start_date: datetime, end_date: datetime):
    self._store.materialize(start_date=start_date, end_date=end_date)
```

Materialization is the process of copying features from the offline store (Parquet files) to the online store (SQLite). In production, this would be scheduled (e.g., hourly via Airflow) to keep the online store fresh.

---

## 5. Training Pipeline — Multi-Architecture Sweeps and MLflow

### Why Multiple Model Architectures?

#### File: `src/training/trainer.py`

There is no universally "best" classification algorithm. The optimal model depends on:

- **Data size and dimensionality** — Logistic regression works well with many features and abundant data. Tree-based models handle interactions and non-linearities better but can overfit small datasets.
- **Feature types** — Trees handle mixed types (numeric + categorical) natively. Linear models require encoding.
- **Interpretability requirements** — Logistic regression coefficients are directly interpretable. Boosted tree feature importances are approximate.
- **Inference latency** — A logistic regression prediction is a single matrix multiply. A 500-tree ensemble requires traversing 500 decision paths.

By training multiple architectures and comparing them on the same holdout set, we let the data decide which model is best rather than making assumptions.

### The Model Registry

Five model families are configured, each with multiple hyperparameter configurations:

```python
MODEL_REGISTRY = {
    "logistic_regression": { ... 4 configs (varying C regularization) },
    "random_forest":       { ... 3 configs (varying depth, trees, min leaf) },
    "gradient_boosting":   { ... 3 configs (varying lr, depth, trees) },
    "lightgbm":            { ... 3 configs (varying lr, leaves, depth) },
    "xgboost":             { ... 2 configs (varying lr, depth, subsampling) },
}
```

This totals 15 configurations. Each one is trained and evaluated independently.

### Cross-Validation

```python
def _cross_validate_model(model, X, y, n_folds=CV_FOLDS):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    scoring = {
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "avg_precision": "average_precision",
    }
    cv_results = cross_validate(model, X, y, cv=skf, scoring=scoring, n_jobs=-1)
```

**Why 5-fold stratified cross-validation?**

A single train/test split gives one estimate of model performance, which has high variance — a lucky or unlucky split can mislead you. K-fold CV trains the model K times, each time holding out a different 1/K of the data for validation. The averaged metrics are a much more reliable estimate of true performance.

**Stratified** means each fold preserves the class ratio. With 3.5% fraud, an unstratified fold could randomly end up with 0 fraud samples in the validation set, making precision/recall undefined.

**`n_jobs=-1`** parallelizes across all available CPU cores. Each fold is independent, so this is embarrassingly parallel.

### MLflow Integration

Every training run logs to MLflow:

```python
with mlflow.start_run(run_name=f"{model_name}_{run_tag}") as run:
    mlflow.set_tag("model_type", model_name)
    mlflow.log_params(params)                    # hyperparameters
    mlflow.log_metrics(cv_metrics)               # cross-validation metrics
    mlflow.log_metrics(test_metrics)             # test set metrics
    mlflow.log_dict(pr_data, "pr_curve.json")    # precision-recall curve
    mlflow.sklearn.log_model(model, "model")     # serialized model artifact
```

MLflow provides:
1. **Experiment comparison** — view all runs side-by-side, sorted by any metric
2. **Reproducibility** — every run records exact parameters and code version
3. **Artifact management** — models are stored with their metadata, downloadable by run ID
4. **Model lineage** — trace which data and parameters produced which model

The tracking URI defaults to a local SQLite database (`mlflow.db`). In production, this would point to a central MLflow server backed by PostgreSQL + S3.

### Metrics Logged

For each model configuration, we log:

| Metric | What It Measures | Why It Matters |
|--------|-----------------|----------------|
| `cv_precision` | Avg precision across folds | How many flagged transactions are actually fraud |
| `cv_recall` | Avg recall across folds | How many actual fraud cases we catch |
| `cv_f1` | Harmonic mean of P and R | Single balanced metric |
| `cv_roc_auc` | Area under ROC curve | Discrimination ability across all thresholds |
| `cv_avg_precision` | Area under PR curve | Better than ROC AUC for imbalanced classes |
| `test_precision` | Precision on holdout set | Final unbiased estimate |
| `test_recall` | Recall on holdout set | Final unbiased estimate |
| `test_f1` | F1 on holdout set | Primary metric for model selection |
| `test_roc_auc` | AUC on holdout set | Final discrimination estimate |
| `training_time_sec` | Wall-clock training time | Inference budget planning |

### Why Average Precision Over ROC AUC?

ROC AUC can be misleadingly optimistic with imbalanced classes. With 96.5% legitimate transactions, a model that predicts "not fraud" for everything achieves 96.5% accuracy and a decent ROC AUC. Average Precision (area under the precision-recall curve) is much more informative because it focuses on the positive (fraud) class performance.

---

## 6. Model Evaluation and Promotion Logic

### Evaluation

#### File: `src/training/evaluator.py`

The evaluator computes a comprehensive metric set for any model on any dataset:

```python
def evaluate_model(model, X, y, dataset_label="eval"):
    metrics = {
        f"{dataset_label}_precision": precision_score(...),
        f"{dataset_label}_recall": recall_score(...),
        f"{dataset_label}_f1": f1_score(...),
        f"{dataset_label}_roc_auc": roc_auc_score(...),
        f"{dataset_label}_avg_precision": average_precision_score(...),
    }
    # Also: TP, FP, TN, FN, and false positive rate
```

The **false positive rate** (`FP / (FP + TN)`) is specifically tracked because in fraud detection, it directly impacts customer experience — legitimate transactions blocked by false positives frustrate customers and cost the business more than most people realize.

### Promotion Comparison

The `compare_models` function implements a multi-gate promotion decision:

**Gate 1: Minimum Precision Bar**
```python
if cand_precision < MIN_PRECISION:  # 0.70
    # REJECT — precision too low
```
A model that flags too many false positives is not deployable regardless of recall. The operations team will stop trusting it.

**Gate 2: Minimum Recall Bar**
```python
if cand_recall < MIN_RECALL:  # 0.55
    # REJECT — missing too much fraud
```
A model that misses too much fraud is not providing value. The 0.55 threshold means we accept missing up to 45% of fraud cases, which sounds bad but is realistic for a first-pass automated system (human review catches more).

**Gate 3: Improvement Over Current Model**
```python
f1_delta = cand_f1 - current_f1
if f1_delta < MIN_F1_IMPROVEMENT:  # 0.005
    # REJECT — not enough improvement to justify a swap
```
Model swaps have risk (new failure modes, edge cases). We require a minimum improvement of 0.5% F1 to justify the risk of changing the production model.

**Gate 4: Regression Guardrails**
```python
if precision_drop > 0.05:  # REJECT
if recall_drop > 0.08:    # REJECT
```
Even if the new model has better F1, we reject it if it achieves that by severely degrading either precision or recall compared to the current model. A model that improves recall by 10% but drops precision by 6% might have better F1 but would flood the ops team with false positives — an unacceptable regression.

### Model Promoter

#### File: `src/pipeline/promoter.py`

The `ModelPromoter` class orchestrates the promotion workflow:

1. Evaluate candidate on the holdout set
2. Load current active model and evaluate it on the *same* holdout set (ensures apples-to-apples comparison)
3. Run `compare_models` to get the verdict
4. **Always save** the candidate to the registry (we keep history even for rejected models)
5. If promoted, update the active model pointer and push metrics to Prometheus

The registry always saves every model because:
- Rejected models are still useful for post-hoc analysis
- If the current model fails catastrophically, you can manually promote an older version
- MLflow run IDs are linked, so you can trace any registry version back to its training run

### Model Registry

#### File: `src/serving/model_loader.py`

The registry uses a filesystem-based versioning scheme:

```
artifacts/registry/
  models/
    v1/
      model.joblib      # serialized scikit-learn model
      metadata.json     # version, name, params, metrics, timestamps
    v2/
      model.joblib
      metadata.json
  active.json           # {"active_version": "v2"}
```

**Atomic promotion** is achieved by writing the `active.json` pointer file. The serving layer reads this file to determine which version to load. A file write on a local filesystem is atomic on all modern operating systems, so there is no window where the pointer is in an inconsistent state.

**Thread-safe loading** uses a `threading.Lock` around the in-memory model reference:

```python
def load_active(self):
    with self._lock:
        if self._active_model is not None:
            return self._active_model, self._active_version, self._active_metadata
```

This prevents race conditions when the serving thread reads the model while the orchestrator thread is promoting a new version.

---

## 7. Drift Detection — Population Stability Index (PSI)

### What Is Data Drift?

#### File: `src/drift/detector.py`

Data drift (also called covariate shift) occurs when the statistical distribution of production data diverges from the distribution the model was trained on. There are several types:

- **Covariate drift**: Input feature distributions change (e.g., average transaction amounts increase)
- **Prior probability drift**: The class balance changes (e.g., fraud rate increases from 3% to 7%)
- **Concept drift**: The relationship between features and labels changes (e.g., a transaction pattern that used to be legitimate is now associated with fraud)

PSI primarily detects covariate drift. Concept drift requires monitoring actual model performance (precision/recall on labeled data), which is harder because production labels are delayed.

### How PSI Works

The Population Stability Index compares two distributions by:

1. **Binning the reference distribution** into N quantile-based buckets
2. **Computing the proportion** of samples in each bucket for both distributions
3. **Calculating the divergence** using a symmetric KL-divergence formula

The formula for PSI:

```
PSI = Σ (P_current(i) - P_reference(i)) × ln(P_current(i) / P_reference(i))
```

where i indexes the bins, and P(i) is the proportion of samples in bin i.

### Why Quantile-Based Binning?

```python
quantiles = np.linspace(0, 100, n_bins + 1)
bin_edges = np.percentile(reference, quantiles)
```

Equal-width bins (e.g., splitting the range [min, max] into 10 equal segments) produce unreliable PSI values when the distribution is skewed. If 95% of values fall in the first bin and the remaining 5% are spread across the other 9, the PSI is dominated by noise in the sparse bins.

Quantile-based binning ensures each bin contains roughly the same number of reference samples, giving stable PSI estimates even for heavily skewed distributions (like transaction amounts).

### Smoothing

```python
eps = 1e-6
ref_pct = (ref_counts + eps) / (ref_counts.sum() + eps * len(ref_counts))
cur_pct = (cur_counts + eps) / (cur_counts.sum() + eps * len(cur_counts))
```

Without smoothing, a bin with zero current samples produces `ln(0)` = negative infinity, and PSI becomes undefined. Adding a small epsilon (1e-6) to all bin counts prevents this while having negligible impact on the result.

### Interpretation Thresholds

| PSI Value | Interpretation | Action |
|-----------|---------------|--------|
| < 0.10 | No significant shift | Continue monitoring |
| 0.10 – 0.25 | Moderate shift | Alert, investigate, monitor closely |
| > 0.25 | Significant shift | Automatic retraining triggered |

These thresholds are industry standard, originally developed in credit risk modeling. They are configurable in `settings.py`:

```python
PSI_THRESHOLD = 0.15        # warning level
PSI_RETRAIN_THRESHOLD = 0.25 # automatic retraining
```

### Aggregate PSI

The detector computes PSI per-feature and then takes the mean as the aggregate score:

```python
agg_psi = float(np.mean(list(feature_psi.values())))
```

The aggregate PSI drives the retraining decision. Per-feature PSI values are logged to Prometheus for dashboard visualization and root-cause analysis — knowing *which* features drifted tells you *what* changed in the real world.

### Feature PSI Summary

The `feature_psi_summary` method returns a ranked DataFrame showing which features drifted most:

```python
def feature_psi_summary(self, current_data):
    # Returns: feature, psi, ref_mean, cur_mean, mean_shift_pct, drifted
```

This is the first thing a data scientist looks at after a drift alert — it tells you whether the shift is in transaction amounts (economic change?), merchant categories (new fraud vector?), or transaction timing (seasonal pattern?).

---

## 8. Serving Layer — FastAPI and Real-Time Inference

### API Design

#### File: `src/serving/app.py`

The serving layer exposes four endpoints:

**`POST /predict`** — Batch fraud scoring

Accepts a list of transactions and returns fraud predictions with probability scores and risk levels:

```json
// Request
{
  "transactions": [
    {
      "tx_amount": 450.0,
      "tx_hour": 3,
      "merchant_category": 3,
      "is_online": 1,
      "is_pin_used": 0,
      ...
    }
  ]
}

// Response
{
  "model_version": "v2",
  "predictions": [
    {
      "is_fraud": true,
      "fraud_probability": 0.8723,
      "risk_level": "high"
    }
  ],
  "latency_ms": 2.34
}
```

**Risk Level Mapping:**
- `high`: probability >= 0.8 — block the transaction, alert fraud team
- `medium`: probability >= 0.4 — allow but flag for review
- `low`: probability < 0.4 — allow

**`GET /health`** — Health check

Returns the current model version and status. Used by load balancers and Kubernetes liveness probes.

**`GET /metrics`** — Prometheus scrape endpoint

Returns all registered Prometheus metrics in the exposition format. This is what Prometheus polls every 5 seconds.

**`GET /model/info`** — Registry introspection

Returns all model versions and which one is active. Useful for debugging and auditing.

### Pydantic Validation

```python
class TransactionInput(BaseModel):
    tx_amount: float = Field(..., gt=0)
    tx_hour: int = Field(..., ge=0, le=23)
    merchant_category: int = Field(..., ge=0, le=7)
    is_chip_used: int = Field(..., ge=0, le=1)
    ...
```

Every field has explicit range validation. This prevents garbage inputs from reaching the model:
- Negative transaction amounts are rejected
- Hours outside 0-23 are rejected
- Binary fields are constrained to {0, 1}

Without input validation, a model can receive `-999` for `tx_amount` and still return a prediction — a prediction that is meaningless because the model never saw negative amounts in training. Validation at the serving boundary is the last line of defense.

### Lifespan Management

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        model, version, meta = registry.load_active()
        PIPELINE_HEALTH.set(1)
    except FileNotFoundError:
        PIPELINE_HEALTH.set(0)
    yield
```

The lifespan context manager loads the model once at startup rather than on every request. This avoids the overhead of deserialization on each prediction call. The model lives in memory for the lifetime of the process.

If no model exists (server started before bootstrap), the health endpoint returns `"no_model"` and the predict endpoint returns HTTP 503 (Service Unavailable). This is the correct behavior — a serving endpoint without a model should not pretend to be healthy.

---

## 9. Monitoring — Prometheus and Grafana

### Prometheus Metrics

#### File: `src/monitoring/metrics.py`

We register metrics on a dedicated `CollectorRegistry` rather than the default global registry. This avoids collisions with Python process metrics that Prometheus client auto-registers, and gives us clean control over exactly what is exported.

**Prediction Metrics:**

| Metric | Type | Labels | Purpose |
|--------|------|--------|---------|
| `ml_predictions_total` | Counter | model_version, predicted_class | Track prediction volume by class |
| `ml_prediction_latency_seconds` | Histogram | model_version | P50/P95/P99 serving latency |
| `ml_prediction_probability` | Histogram | model_version | Distribution of fraud scores |

**Drift Metrics:**

| Metric | Type | Labels | Purpose |
|--------|------|--------|---------|
| `ml_feature_psi` | Gauge | feature_name | Per-feature PSI values |
| `ml_aggregate_psi` | Gauge | — | Overall drift score |
| `ml_drift_alert_level` | Gauge | — | 0=none, 1=warning, 2=critical |

**Model Lifecycle Metrics:**

| Metric | Type | Labels | Purpose |
|--------|------|--------|---------|
| `ml_model_f1_score` | Gauge | model_version, dataset | Current model F1 |
| `ml_model_precision` | Gauge | model_version, dataset | Current model precision |
| `ml_model_recall` | Gauge | model_version, dataset | Current model recall |
| `ml_retrain_events_total` | Counter | trigger_reason | Number of auto-retrains |
| `ml_promotion_events_total` | Counter | — | Number of promotions |

**System Health:**

| Metric | Type | Purpose |
|--------|------|---------|
| `ml_pipeline_health` | Gauge | 1=healthy, 0=degraded |
| `ml_last_drift_check_timestamp` | Gauge | Unix time of last check |
| `ml_last_retrain_timestamp` | Gauge | Unix time of last retrain |

### Why Histograms for Latency?

```python
PREDICTION_LATENCY = Histogram(
    "ml_prediction_latency_seconds",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)
```

Histograms track the distribution of values, not just the current value. Prometheus can compute percentiles from histogram bucket counts:

```promql
histogram_quantile(0.95, rate(ml_prediction_latency_seconds_bucket[5m]))
```

This gives us the P95 latency over the last 5 minutes — the metric that matters for SLAs. A Gauge would only give us the latest single measurement.

The bucket boundaries are chosen based on expected ML inference latency: most predictions should complete in under 25ms; anything over 500ms indicates a problem.

### Grafana Dashboard

#### File: `dashboards/pipeline_overview.json`

The dashboard is organized in four rows:

**Row 1 — Status Tiles:** Pipeline health, active model, total predictions, retrain count, drift alert gauge. This row gives you the at-a-glance answer to "is the pipeline healthy right now?"

**Row 2 — Drift Monitoring:** Aggregate PSI time series (with threshold lines at 0.15 and 0.25) and per-feature PSI breakdowns. When PSI crosses the yellow line, you pay attention. When it crosses the red line, the pipeline is already self-healing.

**Row 3 — Serving Performance:** P50 and P95 prediction latency, and prediction volume split by class (fraud vs. legitimate). A sudden spike in fraud predictions might indicate a real attack — or a drifted model over-triggering.

**Row 4 — Feature Distributions:** Feature means over time. When these lines shift, you are seeing the data drift that PSI quantifies. Useful for root-cause analysis after a drift alert.

### Prometheus Configuration

```yaml
scrape_configs:
  - job_name: "ml_pipeline"
    static_configs:
      - targets: ["pipeline:8000"]
    metrics_path: /metrics
    scrape_interval: 5s
```

The 5-second scrape interval is aggressive but appropriate for an ML system where drift alerts should be visible quickly. For a production deployment with hundreds of targets, you would relax this to 15-30 seconds to reduce Prometheus load.

---

## 10. The Self-Healing Loop — How It All Connects

### Orchestrator

#### File: `src/pipeline/orchestrator.py`

The `PipelineOrchestrator` class is the conductor. It owns instances of every component and coordinates their interactions.

### Bootstrap Sequence (Cold Start)

```
Step 1: generate_training_data(n=15000, fraud_ratio=0.035, seed=42)
        → 15,000 synthetic transactions spanning 30 days
        → validate_schema() enforces types and handles NaN
        → save to data/training_data.parquet

Step 2: feature_store.initialize(raw_data)
        → writes Feast repo config
        → writes entity + feature view definitions
        → applies definitions to Feast registry
        → feature_store.get_historical_features(entity_df)
        → point-in-time join returns 13 features per entity

Step 3: split_data(feature_df)
        → stratified 80/20 split → 12,000 train / 3,000 test
        → compute_feature_stats() for drift baseline
        → run_hyperparameter_sweep(X_train, y_train, X_test, y_test)
        → trains 15 configurations across 5 architectures
        → each config: 5-fold CV + final test evaluation
        → all logged to MLflow
        → results sorted by test_f1

Step 4: generate_evaluation_report(best_model, X_test, y_test)
        → human-readable classification report
        → confusion matrix with TP/FP/TN/FN

Step 5: promoter.attempt_promotion(candidate=best_model, holdout=test_set)
        → evaluate candidate on holdout
        → no current model exists → passes all gates
        → save to registry as v1
        → promote v1 as active
        → push metrics to Prometheus
```

### Self-Healing Cycle

```python
def check_and_heal(self, incoming_data=None):
```

Each iteration:

1. **Generate or receive production data.** In simulation mode, drift intensity increases with each retrain cycle:
   ```python
   drift_intensity = 0.5 + (self._retrain_count * 0.4)
   ```
   First check: intensity=0.5 (mild). After one retrain: 0.9. After two: 1.3. This ensures the pipeline faces increasingly challenging distribution shifts.

2. **Run drift detection.** Compute PSI for each feature against the reference distribution. Push results to Prometheus.

3. **If PSI < retrain threshold:** Log, sleep, check again next iteration.

4. **If PSI >= retrain threshold:** Self-healing activates:

   a. **Combine data:** Merge old training data with new drifted data. This is important — we do not discard old data entirely, because some patterns from the old distribution are still valid. The combined dataset lets the model learn both old and new patterns.

   b. **Focused retraining:** Only sweep the top-performing architectures (LightGBM, XGBoost, Gradient Boosting) rather than all 15 configurations. This is a deliberate optimization — logistic regression and random forest rarely beat boosted methods on tabular data, and retraining should be fast.

   c. **Promote if better:** Run the full promotion gate logic. The new model must beat the current model by at least 0.5% F1 without regressing precision or recall too much.

   d. **Update reference:** If promoted, the drift detector's reference distribution is updated to the new training data. This "resets the clock" — future drift is measured against the post-retrain distribution.

### Why Combine Old and New Data?

Training only on drifted data would cause **catastrophic forgetting** — the model would lose its ability to detect fraud patterns that were present in the original data but not in the drift sample.

Training only on old data would not address the drift — the model would keep making the same mistakes.

Combining both gives the model access to the full pattern space while giving recent patterns more weight (simply because there are more recent samples in the combined dataset if drift batches accumulate).

### Continuous Monitoring Loop

```python
def run_drift_monitor(self, interval_seconds=30, max_iterations=0):
    while self._running:
        result = self.check_and_heal()
        time.sleep(interval_seconds)
```

The loop runs in the main thread while the FastAPI server runs in a daemon thread. This means:
- If the main process is killed (SIGINT/SIGTERM), the signal handler calls `orchestrator.stop()` which sets `self._running = False`, cleanly exiting the loop.
- The API server thread is a daemon, so it dies automatically when the main thread exits.
- There is no risk of the API serving a half-loaded model because the registry uses a lock.

---

## 11. Configuration Reference

### `config/settings.py` — Full Parameter Reference

#### Path Configuration
| Parameter | Default | Purpose |
|-----------|---------|---------|
| `BASE_DIR` | Project root | All paths are relative to this |
| `DATA_DIR` | `{BASE_DIR}/data` | Generated datasets |
| `ARTIFACTS_DIR` | `{BASE_DIR}/artifacts` | Model registry, metadata |
| `FEATURE_REPO_DIR` | `{BASE_DIR}/config/feature_repo` | Feast repository |

#### MLflow
| Parameter | Default | Purpose |
|-----------|---------|---------|
| `MLFLOW_TRACKING_URI` | `sqlite:///mlflow.db` | Experiment tracking database |
| `MLFLOW_EXPERIMENT_NAME` | `fraud_detection_pipeline` | Experiment name in MLflow UI |

#### Training
| Parameter | Default | Purpose |
|-----------|---------|---------|
| `RANDOM_SEED` | 42 | Reproducibility seed |
| `TEST_SPLIT_RATIO` | 0.2 | 80/20 train/test split |
| `CV_FOLDS` | 5 | Number of cross-validation folds |
| `TARGET_COLUMN` | `is_fraud` | Label column name |

#### Drift Detection
| Parameter | Default | Purpose |
|-----------|---------|---------|
| `PSI_THRESHOLD` | 0.15 | Alert level (warning) |
| `PSI_RETRAIN_THRESHOLD` | 0.25 | Auto-retrain trigger |
| `PSI_NUM_BINS` | 10 | Bins for PSI calculation |

#### Model Promotion
| Parameter | Default | Purpose |
|-----------|---------|---------|
| `MIN_PRECISION` | 0.70 | Minimum acceptable precision |
| `MIN_RECALL` | 0.55 | Minimum acceptable recall |
| `MIN_F1_IMPROVEMENT` | 0.005 | Min improvement to justify swap |

#### Data Generation
| Parameter | Default | Purpose |
|-----------|---------|---------|
| `NUM_TRAIN_SAMPLES` | 15,000 | Training dataset size |
| `NUM_DRIFT_SAMPLES` | 3,000 | Drift batch size |
| `FRAUD_RATIO` | 0.035 | Target fraud rate |

---

## 12. Scaling Up — From Laptop to Production

### Current Limitations

The local setup works for development and demonstration, but has several limitations that need to be addressed for production deployment:

1. **Single-process training** — model sweep is sequential across architectures (parallelized within each model via `n_jobs=-1` but not across models)
2. **File-based storage** — Parquet + SQLite works for small data but does not scale
3. **No authentication** — API endpoints are open
4. **No model versioning beyond local filesystem** — no centralized model store
5. **No real streaming data** — drift detection runs on synthetic batches
6. **Single-node serving** — no horizontal scaling or load balancing

### Tier 1: Production-Ready Single Node

**Estimated effort: 1-2 weeks**

These changes make the pipeline deployable on a single production server with proper reliability guarantees.

#### 1.1 Replace SQLite with PostgreSQL

```yaml
# docker-compose.yml addition
postgres:
  image: postgres:16-alpine
  environment:
    POSTGRES_DB: mlpipeline
    POSTGRES_USER: pipeline
    POSTGRES_PASSWORD: ${DB_PASSWORD}
  volumes:
    - pg_data:/var/lib/postgresql/data
  ports:
    - "5432:5432"
```

Update MLflow tracking URI:
```python
MLFLOW_TRACKING_URI = "postgresql://pipeline:${DB_PASSWORD}@postgres:5432/mlpipeline"
```

Update Feast online store:
```yaml
online_store:
  type: postgres
  host: postgres
  port: 5432
  database: mlpipeline
  db_schema: feast
  user: pipeline
  password: ${DB_PASSWORD}
```

**Why:** SQLite does not handle concurrent writes (training + serving + monitoring all hitting the same DB). PostgreSQL handles thousands of concurrent connections, has MVCC (no read locks), and supports proper backup/restore.

#### 1.2 Add Redis for Feature Caching

```yaml
redis:
  image: redis:7-alpine
  ports:
    - "6379:6379"
  command: redis-server --maxmemory 256mb --maxmemory-policy allkeys-lru
```

```python
# In serving/app.py — cache hot features
import redis
cache = redis.Redis(host="redis", port=6379, decode_responses=True)

async def get_features_cached(entity_id: str):
    cached = cache.get(f"features:{entity_id}")
    if cached:
        return json.loads(cached)
    features = feature_store.get_online_features([entity_id])
    cache.setex(f"features:{entity_id}", 300, features.to_json())  # 5 min TTL
    return features
```

**Why:** Feast online store lookups add 5-15ms per request. Redis brings this down to <1ms for frequently seen users. The 5-minute TTL is short enough that feature freshness is maintained.

#### 1.3 API Authentication

```python
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Depends, Security

security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    token = credentials.credentials
    # Validate JWT or API key against your auth service
    if not validate_token(token):
        raise HTTPException(401, "Invalid authentication token")
    return token

@app.post("/predict")
async def predict(request: PredictionRequest, token: str = Depends(verify_token)):
    ...
```

**Why:** Without authentication, anyone who can reach the API can submit predictions or scrape the /metrics endpoint (which reveals model metadata).

#### 1.4 Structured Logging

```python
import structlog

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer(),
    ],
)
logger = structlog.get_logger()

# Usage
logger.info("drift_check_complete",
    aggregate_psi=0.23,
    drifted_features=["tx_amount", "distance_from_home"],
    should_retrain=False,
)
```

**Why:** JSON-structured logs are searchable in log aggregation systems (ELK, Splunk, CloudWatch Logs). Unstructured text logs are nearly impossible to query at scale.

#### 1.5 Health Check Improvements

```python
@app.get("/health")
async def health():
    checks = {
        "model_loaded": registry._active_model is not None,
        "model_version": registry._active_version,
        "last_drift_check_age_seconds": time.time() - last_drift_check_ts,
        "drift_check_stale": (time.time() - last_drift_check_ts) > 120,
    }
    healthy = checks["model_loaded"] and not checks["drift_check_stale"]
    status_code = 200 if healthy else 503
    return JSONResponse(content=checks, status_code=status_code)
```

**Why:** Kubernetes liveness/readiness probes need more than "process is running." If the model is not loaded or drift checks have stopped, the pod should be marked unhealthy and restarted.

---

### Tier 2: Horizontally Scaled Deployment

**Estimated effort: 3-6 weeks**

These changes enable the pipeline to handle production-scale traffic across multiple nodes.

#### 2.1 Kubernetes Deployment

```yaml
# k8s/serving-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fraud-serving
spec:
  replicas: 3
  selector:
    matchLabels:
      app: fraud-serving
  template:
    spec:
      containers:
        - name: serving
          image: fraud-pipeline:latest
          command: ["python", "run_pipeline.py", "--serve"]
          ports:
            - containerPort: 8000
          resources:
            requests:
              memory: "512Mi"
              cpu: "500m"
            limits:
              memory: "1Gi"
              cpu: "1000m"
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 30
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 10
            periodSeconds: 5
---
# Separate deployment for the training/monitoring loop
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fraud-trainer
spec:
  replicas: 1  # only one trainer at a time
  template:
    spec:
      containers:
        - name: trainer
          image: fraud-pipeline:latest
          command: ["python", "run_pipeline.py", "--monitor-interval", "300"]
          resources:
            requests:
              memory: "2Gi"
              cpu: "2000m"
```

**Architecture change:** Sepaafter
        # serialize to bytes
        buffer = io.BytesIO()
        joblib.dump(model, buffer)
        self.s3.put_object(
            Bucket=self.bucket,
            Key=f"{self.prefix}/{version}/model.joblib",
            Body=buffer.getvalue(),
        )

    def promote(self, version):
        # update the active pointer
        self.s3.put_object(
            Bucket=self.bucket,
            Key=f"{self.prefix}/active.json",
            Body=json.dumps({"active_version": version}),
        )
        # notify serving pods to reload
        self._publish_reload_event(version)
```

**Why:** With multiple serving replicas, all pods need access to the same model. A local filesystem is per-pod. S3/GCS is the standard for centralized artifact storage.

**Model reload notification:** After promoting a new model, the trainer publishes an event (via Redis pub/sub, SNS, or a Kubernetes ConfigMap update) that tells serving pods to reload. Without this, serving pods would serve the old model until they restart.

#### 2.3 Kafka/Kinesis for Streaming Data

Replace synthetic drift data with a real streaming pipeline:

```python
from confluent_kafka import Consumer

class StreamingDriftMonitor:
    def __init__(self, topic="transactions", bootstrap_servers="kafka:9092"):
        self.consumer = Consumer({
            "bootstrap.servers": bootstrap_servers,
            "group.id": "drift-monitor",
            "auto.offset.reset": "latest",
        })
        self.consumer.subscribe([topic])
        self.buffer = []

    def collect_batch(self, batch_size=5000, timeout_seconds=300):
        """Accumulate messages until batch is full or timeout."""
        start = time.time()
        while len(self.buffer) < batch_size:
            msg = self.consumer.poll(1.0)
            if msg and not msg.error():
                self.buffer.append(json.loads(msg.value()))
            if time.time() - start > timeout_seconds:
                break

        batch = pd.DataFrame(self.buffer[:batch_size])
        self.buffer = self.buffer[batch_size:]
        return batch
```

**Why:** In production, transactions arrive continuously. Kafka provides durable, ordered, replay-able streams. The drift monitor consumes batches of 5,000 transactions (or whatever accumulates in 5 minutes) and runs PSI against the reference distribution.

#### 2.4 Airflow for Pipeline Orchestration

Replace the Python `while` loop with Airflow DAGs:

```python
# dags/drift_check_dag.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import timedelta

dag = DAG(
    "drift_check",
    schedule_interval=timedelta(minutes=30),
    catchup=False,
)

def check_drift(**context):
    from src.drift.detector import DriftDetector
    # ... load reference, fetch recent data, compute PSI
    return drift_report

def retrain(**context):
    ti = context["task_instance"]
    drift_report = ti.xcom_pull(task_ids="check_drift")
    if drift_report["should_retrain"]:
        # ... run training sweep
        pass

check_task = PythonOperator(task_id="check_drift", python_callable=check_drift, dag=dag)
retrain_task = PythonOperator(task_id="retrain", python_callable=retrain, dag=dag)
check_task >> retrain_task
```

**Why:** Airflow provides retry logic, dependency management, scheduling, logging, and a web UI for monitoring DAG runs. The custom `while` loop works for a single-node demo but lacks:
- Automatic retries on transient failures
- Alerting when a DAG run fails
- Backfill capability (re-run drift checks for historical data)
- Audit trail (who triggered what, when)

---

### Tier 3: Enterprise-Scale ML Platform

**Estimated effort: 2-6 months**

These changes transform the pipeline into a full ML platform supporting multiple models, teams, and regulatory requirements.

#### 3.1 Kubeflow Pipelines for Training

```python
from kfp import dsl, compiler

@dsl.component(base_image="python:3.11")
def train_component(data_path: str, model_name: str) -> str:
    # ... training logic
    return model_version

@dsl.component(base_image="python:3.11")
def evaluate_component(model_version: str, holdout_path: str) -> dict:
    # ... evaluation logic
    return metrics

@dsl.pipeline(name="fraud-retraining")
def retraining_pipeline(data_path: str):
    train_task = train_component(data_path=data_path, model_name="lightgbm")
    eval_task = evaluate_component(
        model_version=train_task.output,
        holdout_path=data_path,
    )
```

**Why:** Kubeflow Pipelines provides:
- GPU scheduling for large models
- Artifact lineage tracking
- Pipeline versioning
- Caching of intermediate results
- Multi-step pipelines with conditional logic

#### 3.2 A/B Testing for Model Promotion

Instead of hard-cutting to the new model, gradually shift traffic:

```python
class ABTestRouter:
    def __init__(self, control_model, treatment_model, treatment_fraction=0.1):
        self.control = control_model
        self.treatment = treatment_model
        self.treatment_fraction = treatment_fraction

    def predict(self, X, request_id):
        # Deterministic bucketing so the same user always gets the same model
        bucket = hash(request_id) % 100
        if bucket < self.treatment_fraction * 100:
            return self.treatment.predict(X), "treatment"
        return self.control.predict(X), "control"
```

Then, after sufficient data is collected:

```python
def evaluate_ab_test(control_metrics, treatment_metrics, min_samples=10000):
    """Statistical significance test for A/B experiment."""
    from scipy.stats import proportions_ztest

    # Compare fraud detection rates
    z_stat, p_value = proportions_ztest(
        [treatment_metrics["tp"], control_metrics["tp"]],
        [treatment_metrics["total"], control_metrics["total"]],
    )
    return {
        "p_value": p_value,
        "significant": p_value < 0.05,
        "treatment_better": treatment_metrics["f1"] > control_metrics["f1"],
    }
```

**Why:** The current promotion logic uses holdout evaluation, which is offline. A/B testing validates the model on real production traffic with real labels. This catches issues that holdout evaluation misses:
- Feature store bugs that only manifest in production
- Latency differences between models affecting user behavior
- Edge cases not represented in the holdout set

#### 3.3 Feature Store Scale-Up (Feast on Cloud)

```yaml
# feature_store.yaml — production configuration
project: fraud_detection
provider: gcp
registry: gs://ml-platform/feast/registry.db
online_store:
  type: redis
  connection_string: redis://redis-cluster:6379
offline_store:
  type: bigquery
  project: my-gcp-project
  dataset: feast_features
```

**Why:** The local Parquet + SQLite setup handles thousands of entities. Production fraud detection needs to serve features for millions of users with sub-10ms latency. Redis as the online store provides this. BigQuery as the offline store handles historical feature retrieval for training at any scale.

#### 3.4 Distributed Training

For models that benefit from large datasets (hundreds of millions of rows):

```python
# Using Dask for distributed preprocessing
import dask.dataframe as dd

ddf = dd.read_parquet("s3://data/transactions/", engine="pyarrow")
X_train = ddf[FEATURE_COLUMNS].compute()

# Using Ray for distributed hyperparameter tuning
from ray import tune
from ray.tune.sklearn import TuneSearchCV

search = TuneSearchCV(
    LGBMClassifier(),
    param_distributions={
        "n_estimators": [100, 200, 400, 800],
        "max_depth": [6, 8, 10, 12],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "num_leaves": [31, 63, 127, 255],
    },
    n_trials=50,
    scoring="f1",
    cv=5,
    refit=True,
    use_gpu=False,
    max_concurrent=8,  # 8 trials in parallel across Ray cluster
)
```

**Why:** With 15,000 samples, training is fast on a single machine. With 100M+ samples, single-node training takes hours. Dask distributes preprocessing across a cluster. Ray Tune distributes hyperparameter search across multiple machines, running 8+ model configurations in parallel.

#### 3.5 Model Explainability

Add SHAP explanations to predictions:

```python
import shap

class ExplainableFraudModel:
    def __init__(self, model):
        self.model = model
        self.explainer = shap.TreeExplainer(model)

    def predict_with_explanation(self, X):
        predictions = self.model.predict_proba(X)[:, 1]
        shap_values = self.explainer.shap_values(X)

        explanations = []
        for i in range(len(X)):
            top_features = sorted(
                zip(X.columns, shap_values[1][i]),
                key=lambda x: abs(x[1]),
                reverse=True,
            )[:5]
            explanations.append({
                "top_contributing_features": [
                    {"feature": f, "impact": round(float(v), 4)}
                    for f, v in top_features
                ]
            })

        return predictions, explanations
```

```json
// Example response
{
  "fraud_probability": 0.87,
  "explanation": {
    "top_contributing_features": [
      {"feature": "tx_amount", "impact": 0.23},
      {"feature": "is_pin_used", "impact": -0.18},
      {"feature": "distance_from_home", "impact": 0.15}
    ]
  }
}
```

**Why:** Regulatory requirements (GDPR Article 22, ECOA in the US) require that automated decisions affecting individuals can be explained. "The model said fraud" is not sufficient — you need to say *why*.

#### 3.6 Shadow Mode Deployment

Before promoting a new model to handle real traffic, run it in shadow mode:

```python
class ShadowRouter:
    """
    Route all traffic to the primary model but also score
    with the shadow model. Log both predictions for comparison.
    Only the primary's prediction is returned to the caller.
    """
    def __init__(self, primary, shadow):
        self.primary = primary
        self.shadow = shadow

    async def predict(self, X):
        primary_pred = self.primary.predict_proba(X)[:, 1]

        # fire-and-forget shadow prediction
        shadow_pred = self.shadow.predict_proba(X)[:, 1]
        log_shadow_comparison(primary_pred, shadow_pred)

        return primary_pred
```

**Why:** Shadow mode lets you observe the new model's behavior on production traffic without any risk. If the shadow model would have made different decisions, you can analyze those cases *before* promoting it. This catches issues that offline evaluation cannot.

#### 3.7 Monitoring Enhancements

**Alerting Rules (Prometheus AlertManager):**

```yaml
groups:
  - name: ml_pipeline
    rules:
      - alert: HighDriftDetected
        expr: ml_aggregate_psi > 0.25
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Data drift detected (PSI={{ $value }})"

      - alert: ModelRetrainingFailed
        expr: increase(ml_retrain_events_total[1h]) > 0 and ml_pipeline_health == 0
        for: 10m
        labels:
          severity: critical
        annotations:
          summary: "Model retraining triggered but pipeline is unhealthy"

      - alert: PredictionLatencyHigh
        expr: histogram_quantile(0.95, rate(ml_prediction_latency_seconds_bucket[5m])) > 0.5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "P95 prediction latency above 500ms"
```

**Why:** Dashboards require someone to look at them. Alerts proactively notify the team. The three alerts above cover the three most critical failure modes:
1. The data has changed and the model may be degrading
2. The self-healing mechanism itself has failed
3. The serving layer is slow (infrastructure issue)

---

### Scaling Summary Matrix

| Aspect | Current (Local) | Tier 1 (Single Node) | Tier 2 (Multi-Node) | Tier 3 (Enterprise) |
|--------|----------------|---------------------|---------------------|---------------------|
| **Storage** | SQLite + Parquet | PostgreSQL | PostgreSQL + S3 | BigQuery + S3 + Redis |
| **Training** | Sequential, local | Sequential, local | Parallel (Airflow) | Distributed (Ray/Kubeflow) |
| **Serving** | Single process | Single process + Gunicorn | K8s replicas + LB | K8s + A/B + Shadow |
| **Features** | Feast (file) | Feast (Postgres) | Feast (Redis online) | Feast (Redis + BigQuery) |
| **Monitoring** | Prometheus + Grafana | + Structured logs | + AlertManager | + SHAP + Audit trail |
| **Data** | Synthetic batch | Real data, batch | Kafka streaming | Kafka + CDC |
| **Drift Check** | While loop | Cron | Airflow DAG | Airflow + streaming |
| **Auth** | None | API key | OAuth2/JWT | mTLS + RBAC |
| **Scale** | ~100 req/s | ~1K req/s | ~50K req/s | ~500K+ req/s |

---

*This pipeline demonstrates the core principles of MLOps: automated training, continuous monitoring, drift detection, and autonomous self-healing. The local implementation is a faithful representation of the production architecture — every component has a direct analog in a scaled deployment. The gap between this codebase and production is infrastructure and scale, not architecture.*
