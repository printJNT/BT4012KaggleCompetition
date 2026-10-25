"""
BT4012 Kaggle Competition — Fraud Detection on Bitcoin Transaction Graph
Step-by-step pipeline built incrementally.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from lightgbm import LGBMClassifier
import lightgbm as lgb
import networkx as nx
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# STEP 1: DATA LOADING & SANITY CHECKS
# ============================================================

DATA_DIR = r"C:\National University of Singapore\OneDrive - National University of Singapore\NUS\Year 4 Semester 1\BT4012\Kaggle Competition\BT4012KaggleCompetitionDataset2026"

print("=" * 60)
print("STEP 1: Loading datasets...")
print("=" * 60)

train_df = pd.read_csv(f"{DATA_DIR}\\train.csv")
test_df = pd.read_csv(f"{DATA_DIR}\\test.csv")
edgelist_df = pd.read_csv(f"{DATA_DIR}\\txs_edgelist.csv")
sample_sub = pd.read_csv(f"{DATA_DIR}\\sample_submission.csv")

print(f"\ntrain.csv    : {train_df.shape[0]:>10,} rows × {train_df.shape[1]:>4} cols")
print(f"test.csv     : {test_df.shape[0]:>10,} rows × {test_df.shape[1]:>4} cols")
print(f"edgelist.csv : {edgelist_df.shape[0]:>10,} rows × {edgelist_df.shape[1]:>4} cols")
print(f"sample_sub   : {sample_sub.shape[0]:>10,} rows × {sample_sub.shape[1]:>4} cols")

# --- Label distribution ---
print(f"\n--- Train Label Distribution ---")
print(train_df["label"].value_counts(dropna=False))

# --- Time step ranges ---
print(f"\n--- Time Step Ranges ---")
print(f"Train time_step : {train_df['time_step'].min()} to {train_df['time_step'].max()}")
print(f"Test  time_step : {test_df['time_step'].min()} to {test_df['time_step'].max()}")

# Confirm test is strictly AFTER train
assert test_df["time_step"].min() > train_df["time_step"].max(), \
    "ERROR: Test time steps overlap with train!"
print("✓ Confirmed: test time steps are strictly after train time steps.")

# --- Missing values ---
train_missing = train_df.isnull().sum().sum()
test_missing = test_df.isnull().sum().sum()
print(f"\n--- Missing Values ---")
print(f"Train missing values: {train_missing}")
print(f"Test  missing values: {test_missing}")

# --- Sample submission format ---
print(f"\n--- Sample Submission Format ---")
print(sample_sub.head())
print(f"\nColumns: {list(sample_sub.columns)}")

# --- Edge list sample ---
print(f"\n--- Edge List Sample ---")
print(edgelist_df.head())
print(f"Unique source nodes (txId1): {edgelist_df['txId1'].nunique():,}")
print(f"Unique target nodes (txId2): {edgelist_df['txId2'].nunique():,}")

print(f"\n{'=' * 60}")
print("STEP 1 COMPLETE ✓")
print(f"{'=' * 60}")


# ============================================================
# STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================
print(f"\n{'=' * 60}")
print("STEP 2: Exploratory Data Analysis (EDA)...")
print(f"{'=' * 60}")

# Identify feature columns
feature_cols = [c for c in train_df.columns if c.startswith("feat_")]
print(f"\nFeature columns count: {len(feature_cols)} (from {feature_cols[0]} to {feature_cols[-1]})")

# Check temporal distribution of labels (licit vs illicit vs unknown per time_step)
print("\n--- Label Distribution Across Time Steps (first 5 and last 5) ---")
step_summary = pd.DataFrame({
    "total_txs": train_df.groupby("time_step").size(),
    "licit": train_df[train_df["label"].isin([0, 0.0, "0"])].groupby("time_step").size(),
    "illicit": train_df[train_df["label"].isin([1, 1.0, "1"])].groupby("time_step").size(),
    "unlabeled": train_df[train_df["label"].isna()].groupby("time_step").size(),
}).fillna(0).astype(int)

step_summary["labeled_total"] = step_summary["licit"] + step_summary["illicit"]
step_summary["illicit_pct"] = (step_summary["illicit"] / step_summary["labeled_total"] * 100).round(2)

print(pd.concat([step_summary.head(5), step_summary.tail(5)]))

# Quick summary statistics across features
print("\n--- Feature Stats Sample (feat_1 to feat_5) ---")
print(train_df[feature_cols[:5]].describe().round(4))

print(f"\n{'=' * 60}")
print("STEP 2 COMPLETE ✓")
print(f"{'=' * 60}")


# ============================================================
# STEP 3: DATA PREPROCESSING
# ============================================================
print(f"\n{'=' * 60}")
print("STEP 3: Data Preprocessing...")
print(f"{'=' * 60}")

# 1. Guarantee submission row alignment for test set
test_df = test_df.assign(original_order=np.arange(len(test_df)))
print(f"\nTagged test_df with 'original_order' (0 to {len(test_df) - 1})")

# 2. Detrend time-proxy features within each time_step to mitigate out-of-distribution drift
DRIFT_COLS = ["feat_100", "feat_101", "feat_103", "feat_136", "feat_139", "feat_137"]
detrended_feature_cols = [f"{c}_detrended" for c in DRIFT_COLS]

# Drop existing detrended columns if re-running cell
train_df = train_df.drop(columns=[c for c in detrended_feature_cols if c in train_df.columns])
test_df = test_df.drop(columns=[c for c in detrended_feature_cols if c in test_df.columns])

all_drift = pd.concat([train_df[["time_step"] + DRIFT_COLS], test_df[["time_step"] + DRIFT_COLS]], ignore_index=True)
medians = all_drift.groupby("time_step")[DRIFT_COLS].transform("median")
detrended_df = all_drift[DRIFT_COLS] - medians
detrended_df.columns = detrended_feature_cols

train_df = pd.concat([train_df, detrended_df.iloc[:len(train_df)]], axis=1)
test_df = pd.concat([test_df, detrended_df.iloc[len(train_df):].reset_index(drop=True)], axis=1)
print(f"Computed {len(detrended_feature_cols)} detrended features: {detrended_feature_cols}")

# 3. Extract labeled subset for supervised training
labeled_mask = train_df["label"].isin([0, 1, 0.0, 1.0])
labeled_train_df = train_df[labeled_mask].copy()

# 4. Cast label to integer target (0 = licit, 1 = illicit)
labeled_train_df["target"] = labeled_train_df["label"].astype(int)

print(f"\nOriginal train rows : {len(train_df):,}")
print(f"Labeled train rows  : {len(labeled_train_df):,} ({len(labeled_train_df)/len(train_df)*100:.2f}%)")
print(f"Unlabeled train rows: {(~labeled_mask).sum():,} ({(~labeled_mask).sum()/len(train_df)*100:.2f}%)")

print(f"\nTarget class balance in labeled subset:")
print(labeled_train_df["target"].value_counts(normalize=False))
print(f"Illicit proportion: {labeled_train_df['target'].mean() * 100:.2f}%")

print(f"\n{'=' * 60}")
print("STEP 3 COMPLETE ✓")
print(f"{'=' * 60}")


# ============================================================
# STEP 4: GRAPH FEATURE ENGINEERING (DEEPER TOPOLOGIES)
# ============================================================
print(f"\n{'=' * 60}")
print("STEP 4: Graph Feature Engineering (Deeper Topologies)...")
print(f"{'=' * 60}")

# 1. 1-Hop Degree Metrics (Calculated over full graph across all time steps)
in_degrees = edgelist_df.groupby("txId2").size().rename("graph_in_degree")
out_degrees = edgelist_df.groupby("txId1").size().rename("graph_out_degree")

print(f"\nUnique transactions with in-edges : {len(in_degrees):,}")
print(f"Unique transactions with out-edges: {len(out_degrees):,}")

# 2. 2-Hop Degree Aggregations (Predecessor & Successor Flow Patterns)
print("Computing 2-hop predecessor and successor degree flows...")
edges_deg = edgelist_df.copy()
edges_deg["src_in_deg"] = edges_deg["txId1"].map(in_degrees).fillna(0)
edges_deg["src_out_deg"] = edges_deg["txId1"].map(out_degrees).fillna(0)
edges_deg["dst_in_deg"] = edges_deg["txId2"].map(in_degrees).fillna(0)
edges_deg["dst_out_deg"] = edges_deg["txId2"].map(out_degrees).fillna(0)

# Predecessors of txId2 (who funded this transaction?)
pred_stats = edges_deg.groupby("txId2").agg(
    pred_in_deg_mean=("src_in_deg", "mean"),
    pred_in_deg_max=("src_in_deg", "max"),
    pred_out_deg_mean=("src_out_deg", "mean"),
    pred_out_deg_max=("src_out_deg", "max"),
)

# Successors of txId1 (where did funds from this transaction go?)
succ_stats = edges_deg.groupby("txId1").agg(
    succ_in_deg_mean=("dst_in_deg", "mean"),
    succ_in_deg_max=("dst_in_deg", "max"),
    succ_out_deg_mean=("dst_out_deg", "mean"),
    succ_out_deg_max=("dst_out_deg", "max"),
)

# 3. Global Graph Topology: PageRank and Weakly Connected Components (WCC)
print("Computing PageRank and Weakly Connected Components...")
G = nx.from_pandas_edgelist(edgelist_df, source="txId1", target="txId2", create_using=nx.DiGraph())

pr_dict = nx.pagerank(G, alpha=0.85, max_iter=100, tol=1e-5)
pr_series = pd.Series(pr_dict, name="graph_pagerank")

wcc_map = {}
for comp in nx.weakly_connected_components(G):
    c_size = len(comp)
    for node in comp:
        wcc_map[node] = c_size
wcc_series = pd.Series(wcc_map, name="graph_wcc_size")

graph_feature_cols = [
    "graph_in_degree", "graph_out_degree", "graph_degree_ratio", "graph_net_flow",
    "graph_pagerank", "graph_wcc_size",
    "pred_in_deg_mean", "pred_in_deg_max", "pred_out_deg_mean", "pred_out_deg_max",
    "succ_in_deg_mean", "succ_in_deg_max", "succ_out_deg_mean", "succ_out_deg_max"
]

# 4. Merge all 14 graph metrics cleanly into labeled_train_df and test_df
def add_graph_features(df):
    # If graph features were already added in a previous run, drop them to prevent duplicates
    existing_cols = [c for c in df.columns if c in graph_feature_cols]
    if existing_cols:
        df = df.drop(columns=existing_cols)
    
    in_deg = df["txId"].map(in_degrees).fillna(0).astype(int)
    out_deg = df["txId"].map(out_degrees).fillna(0).astype(int)
    deg_ratio = (in_deg + 1) / (out_deg + 1)
    net_flow = out_deg - in_deg
    pr = df["txId"].map(pr_series).fillna(0.0)
    wcc = df["txId"].map(wcc_series).fillna(1).astype(int)
    
    # 1-hop + global features
    base_graph_df = pd.DataFrame({
        "graph_in_degree": in_deg,
        "graph_out_degree": out_deg,
        "graph_degree_ratio": deg_ratio,
        "graph_net_flow": net_flow,
        "graph_pagerank": pr,
        "graph_wcc_size": wcc,
    }, index=df.index)
    
    # Merge 2-hop predecessor & successor stats
    df_merged = pd.concat([df, base_graph_df], axis=1)
    df_merged = df_merged.merge(pred_stats, left_on="txId", right_index=True, how="left")
    df_merged = df_merged.merge(succ_stats, left_on="txId", right_index=True, how="left")
    
    # Fill isolated/leaf nodes (no 2-hop edges) with 0
    stat_cols = list(pred_stats.columns) + list(succ_stats.columns)
    df_merged[stat_cols] = df_merged[stat_cols].fillna(0.0)
    
    # Guard against any duplicate column names from notebook re-runs
    df_merged = df_merged.loc[:, ~df_merged.columns.duplicated()]
    return df_merged.copy()

print("\nMerging 14 graph features into labeled_train_df and test_df...")
labeled_train_df = add_graph_features(labeled_train_df)
test_df = add_graph_features(test_df)
print(f"\nAdded {len(graph_feature_cols)} graph features: {graph_feature_cols}")

# Quick sanity check on the new features
print("\n--- Graph Features Summary (Labeled Train) ---")
print(labeled_train_df[graph_feature_cols].describe().round(4))

print("\n--- Graph Features by Target (Licit vs Illicit Means) ---")
print(labeled_train_df.groupby("target")[graph_feature_cols].mean().round(4).T)

# 5. Neighbor Ego-Network Features (Aggregating top signals across in/out edges + net flow diff)
print("\nComputing 1-hop neighbor ego-network feature aggregations...")
TOP_SIGNALS = ['feat_1', 'feat_41', 'feat_47']
all_nodes_features = pd.concat([
    train_df[['txId'] + TOP_SIGNALS], 
    test_df[['txId'] + TOP_SIGNALS]
]).drop_duplicates('txId').set_index('txId')

edges_feats = edgelist_df.merge(all_nodes_features, left_on='txId1', right_index=True)
pred_feats = edges_feats.groupby('txId2')[TOP_SIGNALS].agg(['mean', 'max'])
pred_feats.columns = [f"pred_{col}_{stat}" for col, stat in pred_feats.columns]

edges_feats2 = edgelist_df.merge(all_nodes_features, left_on='txId2', right_index=True)
succ_feats = edges_feats2.groupby('txId1')[TOP_SIGNALS].agg(['mean', 'max'])
succ_feats.columns = [f"succ_{col}_{stat}" for col, stat in succ_feats.columns]

nbr_stats = pred_feats.join(succ_feats, how='outer').fillna(0.0)
neighbor_feature_cols = nbr_stats.columns.tolist()

# Merge neighbor stats into labeled_train_df and test_df
existing_nbr_cols = [c for c in labeled_train_df.columns if c in neighbor_feature_cols or c == 'flow_amount_diff']
if existing_nbr_cols:
    labeled_train_df = labeled_train_df.drop(columns=existing_nbr_cols)
    test_df = test_df.drop(columns=[c for c in test_df.columns if c in existing_nbr_cols])

labeled_train_df = labeled_train_df.merge(nbr_stats, left_on='txId', right_index=True, how='left')
test_df = test_df.merge(nbr_stats, left_on='txId', right_index=True, how='left')
labeled_train_df[neighbor_feature_cols] = labeled_train_df[neighbor_feature_cols].fillna(0.0)
test_df[neighbor_feature_cols] = test_df[neighbor_feature_cols].fillna(0.0)

# Flow differential: transaction amount minus mean amount of inputs
labeled_train_df['flow_amount_diff'] = labeled_train_df['feat_1'] - labeled_train_df['pred_feat_1_mean']
test_df['flow_amount_diff'] = test_df['feat_1'] - test_df['pred_feat_1_mean']
neighbor_feature_cols.append('flow_amount_diff')

print(f"Added {len(neighbor_feature_cols)} neighbor aggregation features: {neighbor_feature_cols}")

print(f"\n{'=' * 60}")
print("STEP 4 COMPLETE ✓")
print(f"{'=' * 60}")


# ============================================================
# STEP 5: TEMPORAL TRAIN / VALIDATION SPLIT
# ============================================================
print(f"\n{'=' * 60}")
print("STEP 5: Temporal Train/Validation Split...")
print(f"{'=' * 60}")

# 1. Define model feature columns as an explicit whitelist:
#    165 raw features + 6 detrended + 14 graph topologies + 13 neighbor ego-network = 198 features.
model_feature_cols = feature_cols + detrended_feature_cols + graph_feature_cols + neighbor_feature_cols
print(f"\nModel feature count: {len(model_feature_cols)} "
      f"({len(feature_cols)} raw + {len(detrended_feature_cols)} detrended + {len(graph_feature_cols)} graph + {len(neighbor_feature_cols)} neighbor)")

# Every model feature must exist in the test set too
missing_in_test = [c for c in model_feature_cols if c not in test_df.columns]
assert not missing_in_test, f"Features missing from test_df: {missing_in_test}"

# 2. Split by time_step: roughly the earliest 80% of labeled rows train,
#    the latest 20% validate. Never shuffle — test is strictly in the future.
split_step = labeled_train_df["time_step"].quantile(0.8)
trn_mask = labeled_train_df["time_step"] <= split_step
val_mask = labeled_train_df["time_step"] > split_step

X_train = labeled_train_df.loc[trn_mask, model_feature_cols]
y_train = labeled_train_df.loc[trn_mask, "target"]
X_val = labeled_train_df.loc[val_mask, model_feature_cols]
y_val = labeled_train_df.loc[val_mask, "target"]

trn_steps = labeled_train_df.loc[trn_mask, "time_step"]
val_steps = labeled_train_df.loc[val_mask, "time_step"]
print(f"\nSplit step (80% quantile of labeled time_step): {split_step}")
print(f"Train: time_step {trn_steps.min()}-{trn_steps.max()} | "
      f"{len(X_train):,} rows ({len(X_train)/len(labeled_train_df)*100:.1f}%) | "
      f"illicit {int(y_train.sum()):,} ({y_train.mean()*100:.2f}%)")
print(f"Val  : time_step {val_steps.min()}-{val_steps.max()} | "
      f"{len(X_val):,} rows ({len(X_val)/len(labeled_train_df)*100:.1f}%) | "
      f"illicit {int(y_val.sum()):,} ({y_val.mean()*100:.2f}%)")

# Sanity: no time step appears in both folds
assert trn_steps.max() < val_steps.min(), "Train and validation time steps overlap!"
print("✓ Confirmed: no time-step overlap between train and validation.")

print(f"\n{'=' * 60}")
print("STEP 5 COMPLETE ✓")
print(f"{'=' * 60}")


# ============================================================
# STEP 6: BASELINE MODEL TRAINING (LightGBM)
# ============================================================
print(f"\n{'=' * 60}")
print("STEP 6: Baseline LightGBM Training...")
print(f"{'=' * 60}")

SEED = 42

# 1. Class imbalance weight, computed on the TRAINING fold only
#    (using validation labels here would leak information).
n_licit = int((y_train == 0).sum())
n_illicit = int((y_train == 1).sum())
scale_pos_weight = n_licit / n_illicit
print(f"\nTrain fold: licit={n_licit:,}, illicit={n_illicit:,} "
      f"-> scale_pos_weight = {scale_pos_weight:.4f}")

# 2. Model definition
#    - metric="auc": the competition metric, and the only metric early stopping watches
#    - subsample_freq=1: required, otherwise LightGBM silently ignores `subsample`
#    - deterministic + force_row_wise: reproducible results across runs
model = LGBMClassifier(
    objective="binary",
    metric="auc",
    n_estimators=1200,
    learning_rate=0.03,
    max_depth=6,
    num_leaves=31,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=SEED,
    deterministic=True,
    force_row_wise=True,
    verbose=-1,
)

# 3. Fit on steps 1-28, monitor AUC on steps 29-35.
#    Early stopping halts when validation AUC hasn't improved for 100 rounds.
#    Caveat: because the validation set also picks the stopping round,
#    the validation AUC reported in Step 7 is slightly optimistic.
print("\nTraining (validation AUC logged every 100 rounds)...")
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[
        lgb.early_stopping(stopping_rounds=100, verbose=True),
        lgb.log_evaluation(period=100),
    ],
)

# 4. Record the best round (reused in Step 8 if we refit on all labeled data)
best_iteration = model.best_iteration_
best_val_auc = model.best_score_["valid_0"]["auc"]
print(f"\nBest iteration : {best_iteration}")
print(f"Best val AUC   : {best_val_auc:.5f}")

# 5. Which features does the model rely on most? (gain = total loss reduction)
importance = pd.Series(
    model.booster_.feature_importance(importance_type="gain"),
    index=model_feature_cols,
).sort_values(ascending=False)
print("\n--- Top 15 Features by Gain ---")
print((importance.head(15) / importance.sum() * 100).round(2).rename("gain_%"))
print("\nGraph feature ranks (out of {}):".format(len(model_feature_cols)))
ranks = importance.rank(ascending=False).astype(int)
print(ranks[graph_feature_cols].sort_values())

print(f"\n{'=' * 60}")
print("STEP 6 COMPLETE ✓")
print(f"{'=' * 60}")


# ============================================================
# STEP 7: VALIDATION & METRICS
# ============================================================
print(f"\n{'=' * 60}")
print("STEP 7: Validation Evaluation & Metrics...")
print(f"{'=' * 60}")

# 1. Predict continuous probabilities on the validation fold (steps 29-35)
val_probs = model.predict_proba(X_val)[:, 1]
val_auc = roc_auc_score(y_val, val_probs)

print(f"\n==================================================")
print(f"OFFLINE VALIDATION ROC-AUC: {val_auc:.5f}")
print(f"==================================================")

# 2. Evaluate performance across different decision thresholds (AML trade-off analysis)
#    Lower threshold -> higher Recall (catches more fraud, more false alarms)
#    Higher threshold -> higher Precision (fewer false alarms, misses subtle fraud)
for threshold in [0.3, 0.5, 0.7]:
    preds_binary = (val_probs >= threshold).astype(int)
    print(f"\n--- Classification Metrics @ Decision Threshold = {threshold} ---")
    print(classification_report(y_val, preds_binary, digits=4, target_names=["Licit (0)", "Illicit (1)"]))

print(f"\n{'=' * 60}")
print("STEP 7 COMPLETE ✓")
print(f"{'=' * 60}")


# ============================================================
# STEP 8: RETRAIN ON FULL LABELED DATA & GENERATE SUBMISSION
# ============================================================
print(f"\n{'=' * 60}")
print("STEP 8: Retraining on Full Labeled Data & Generating Submission...")
print(f"{'=' * 60}")

# 1. Prepare full labeled dataset (Steps 1 to 35)
X_full = labeled_train_df[model_feature_cols]
y_full = labeled_train_df["target"]

n_licit_full = int((y_full == 0).sum())
n_illicit_full = int((y_full == 1).sum())
scale_pos_weight_full = n_licit_full / n_illicit_full

print(f"\nFull Labeled Data (Steps 1-35):")
print(f"  Rows          : {len(X_full):,}")
print(f"  Licit (0)     : {n_licit_full:,}")
print(f"  Illicit (1)   : {n_illicit_full:,}")
print(f"  Scale Weight  : {scale_pos_weight_full:.4f}")
print(f"  Trees (rounds): {best_iteration} (optimal round count from Step 6)")

# 2. Train final production ensemble on 100% of labeled data (5-Seed Bagged Production Ensemble)
SEEDS = [42, 123, 456, 789, 2026]
print(f"\nTraining 5-seed bagged production ensemble on steps 1-35 (seeds: {SEEDS})...")

# 3. Ensure test_df is strictly in its original order
test_df = test_df.sort_values("original_order").reset_index(drop=True)

test_preds_list = []
for i, seed in enumerate(SEEDS, 1):
    print(f"  [Seed {seed}] Training model {i}/{len(SEEDS)}...")
    m = LGBMClassifier(
        objective="binary",
        metric="auc",
        n_estimators=best_iteration,
        learning_rate=0.03,
        max_depth=6,
        num_leaves=31,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight_full,
        random_state=seed,
        deterministic=True,
        force_row_wise=True,
        verbose=-1,
    )
    m.fit(X_full, y_full)
    test_preds_list.append(m.predict_proba(test_df[model_feature_cols])[:, 1])

# 4. Average probabilities across the 5 seeds to produce final ensemble predictions
test_probs = np.mean(test_preds_list, axis=0)
print("✓ 5-seed bagged ensemble training and prediction complete.")

# 5. Format into the exact competition submission schema (index, target)
submission_df = pd.DataFrame({
    "index": test_df["original_order"],
    "target": test_probs
})

# 6. Automated sanity guardrails before exporting
assert len(submission_df) == len(sample_sub), f"Row count mismatch! Expected {len(sample_sub)}, got {len(submission_df)}"
assert list(submission_df.columns) == ["index", "target"], f"Column mismatch! Expected ['index', 'target'], got {list(submission_df.columns)}"
assert (submission_df["target"].min() >= 0.0) and (submission_df["target"].max() <= 1.0), "Target probabilities out of [0.0, 1.0] bounds!"
assert submission_df["target"].isna().sum() == 0, "Submission contains NaN predictions!"
assert (submission_df["index"].values == np.arange(len(submission_df))).all(), "Index sequence is non-consecutive or misaligned!"

# 7. Export to submission.csv
SUBMISSION_PATH = "submission.csv"
submission_df.to_csv(SUBMISSION_PATH, index=False)

print(f"\n✓ Updated submission successfully saved to: {SUBMISSION_PATH}")
print(f"\n--- First 5 rows of submission.csv ---")
print(submission_df.head())
print(f"\n--- Prediction Distribution Summary (Full-Trained Model) ---")
print(submission_df["target"].describe().round(4))

print(f"\n{'=' * 60}")
print("STEP 8 COMPLETE ✓ (Ready for Kaggle Upload!)")
print(f"{'=' * 60}")




