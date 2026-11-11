# -*- coding: utf-8 -*-
"""
evaluate.py
------------
Evaluation module separating development from test (Review Item 2).

Key Features:
- 30% Development Split / 70% Held-Out Test Split (stratified).
- Tunes weights (w_spread, w_polysemy, w_diversity) and operating threshold tau*
  strictly on the Dev split.
- Evaluates ONCE on the held-out Test split.
- Non-parametric bootstrapping (1,000 resamples) for 95% Confidence Intervals.
- Precision-Recall curve generation and PR-AUC computation (with plot).
- Saves:
    * sample_output/test_evaluation_results.csv
    * sample_output/test_evaluation_summary.txt
    * sample_output/precision_recall_curve.png
    * sample_output/precision_recall_curve_data.csv
"""

import os
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    classification_report,
    confusion_matrix,
)

from ambiguity_framework import AmbiguityPipeline, AmbiguityScorer, classify_ambiguity


def prf1(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t and p)
    fp = sum(1 for t, p in zip(y_true, y_pred) if not t and p)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t and not p)
    tn = sum(1 for t, p in zip(y_true, y_pred) if not t and not p)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=precision, recall=recall, f1=f1, accuracy=accuracy)


def bootstrap_ci(y_true, y_pred, y_true_type=None, y_pred_type=None, n_bootstraps=1000, alpha=0.05, seed=42):
    """Computes 95% confidence intervals via non-parametric bootstrapping."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    precisions, recalls, f1s, accuracies, type_accs = [], [], [], [], []

    for _ in range(n_bootstraps):
        idxs = rng.choice(n, size=n, replace=True)
        sample_yt = [y_true[i] for i in idxs]
        sample_yp = [y_pred[i] for i in idxs]
        m = prf1(sample_yt, sample_yp)
        precisions.append(m["precision"])
        recalls.append(m["recall"])
        f1s.append(m["f1"])
        accuracies.append(m["accuracy"])

        if y_true_type is not None and y_pred_type is not None:
            sample_tt = [y_true_type[i] for i in idxs]
            sample_tp = [y_pred_type[i] for i in idxs]
            ambig_idxs = [i for i, t in enumerate(sample_tt) if t != "None"]
            if ambig_idxs:
                c = sum(1 for i in ambig_idxs if sample_tt[i] == sample_tp[i])
                type_accs.append(c / len(ambig_idxs))
            else:
                type_accs.append(0.0)

    lower_p = 100 * (alpha / 2)
    upper_p = 100 * (1 - alpha / 2)

    return {
        "accuracy_ci": (np.percentile(accuracies, lower_p), np.percentile(accuracies, upper_p)),
        "precision_ci": (np.percentile(precisions, lower_p), np.percentile(precisions, upper_p)),
        "recall_ci": (np.percentile(recalls, lower_p), np.percentile(recalls, upper_p)),
        "f1_ci": (np.percentile(f1s, lower_p), np.percentile(f1s, upper_p)),
        "type_acc_ci": (np.percentile(type_accs, lower_p), np.percentile(type_accs, upper_p)) if type_accs else (0.0, 0.0),
    }


def extract_features_and_score(pipeline, items_df):
    """Runs pipeline on a dataframe of items and returns structured feature rows."""
    results = []
    for _, row in items_df.iterrows():
        sent = row["requirement_sentence"]
        word = row["target_word"].lower()
        gold_lbl = row["consensus_label"]
        gold_is_ambig = (gold_lbl != "None")

        doc = pipeline.preprocessor.process(sent)
        cands = pipeline.candidate_detector.detect(doc)
        cand = next((c for c in cands if c.lemma == word or c.surface_form.lower() == word), None)

        if cand is None:
            # Word not detected as candidate
            results.append({
                "item_id": row["item_id"],
                "word": word,
                "sentence": sent,
                "domain": row.get("target_domain", "General"),
                "gold_label": gold_lbl,
                "gold_is_ambig": gold_is_ambig,
                "spread": 0.0,
                "baseline_polysemy": 0.0,
                "domain_diversity": 0.0,
                "ambiguity_score": 0.0,
                "predicted_type": "None",
                "ranked_senses": []
            })
            continue

        sent_vec = pipeline.representer.transform([sent])
        (
            senses,
            spread,
            baseline_poly,
            diversity,
            ambig_score,
            sigma_star,
        ) = pipeline.scorer.score(sent_vec, cand, pipeline.representer, pipeline.domain_identifier)

        results.append({
            "item_id": row["item_id"],
            "word": word,
            "sentence": sent,
            "domain": row.get("target_domain", "General"),
            "gold_label": gold_lbl,
            "gold_is_ambig": gold_is_ambig,
            "spread": spread,
            "baseline_polysemy": baseline_poly,
            "domain_diversity": diversity,
            "ambiguity_score": ambig_score,
            "sigma_star": sigma_star,
            "ranked_senses": senses
        })
    return pd.DataFrame(results)


def tune_dev_split(dev_features_df):
    """
    Learns optimal weights via logistic regression and sweeps threshold tau
    to maximize F1 on the development split (30%).
    """
    print("\n--- Tuning on Development Split (30%, N = {}) ---".format(len(dev_features_df)))

    X = dev_features_df[["spread", "baseline_polysemy", "domain_diversity"]].values
    y = dev_features_df["gold_is_ambig"].values.astype(int)

    # Fit Scorer weights on Dev
    scorer = AmbiguityScorer()
    scorer.fit_weights(X, y)
    print(f"Learned Feature Weights (Logistic Regression on Dev):")
    print(f"  w_spread (Spread):           {scorer.w_spread:.3f}")
    print(f"  w_polysemy (Base Polysemy):  {scorer.w_polysemy:.3f}")
    print(f"  w_diversity (Domain Div):    {scorer.w_diversity:.3f}")

    # Recompute scores using learned weights
    dev_scores = (
        scorer.w_spread * dev_features_df["spread"]
        + scorer.w_polysemy * dev_features_df["baseline_polysemy"]
        + scorer.w_diversity * dev_features_df["domain_diversity"]
    ).values

    # Sweep threshold tau to maximize F1
    best_tau, best_f1 = 0.35, -1.0
    for tau in np.arange(0.15, 0.85, 0.02):
        preds = dev_scores >= tau
        m = prf1(y, preds)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_tau = float(tau)

    print(f"Optimal Threshold on Dev (tau*): {best_tau:.2f} (Dev F1 = {best_f1:.3f})")
    return scorer, best_tau


def evaluate_pipeline():
    print("=" * 70)
    print("SEPARATING DEVELOPMENT FROM TEST EVALUATION (Review Item 2)")
    print("=" * 70)

    # 1. Load 250-item Gold Standard
    if not os.path.exists("gold_standard_250.csv"):
        from build_gold_standard import generate_gold_standard
        df_all = generate_gold_standard()
    else:
        df_all = pd.read_csv("gold_standard_250.csv", keep_default_na=False)

    # 2. Stratified 30% Dev / 70% Test Split
    train_dev_df, test_df = train_test_split(
        df_all,
        test_size=0.70,
        random_state=42,
        stratify=df_all["consensus_label"]
    )
    print(f"Total benchmark:    {len(df_all)} items")
    print(f"Development split:  {len(train_dev_df)} items (30.0%)")
    print(f"Held-out test split:{len(test_df)} items (70.0%)")

    # 3. Fit Pipeline Vocabulary on Dev Corpus
    pipeline = AmbiguityPipeline(ambiguity_threshold=0.35)
    pipeline.fit_vocabulary(train_dev_df["requirement_sentence"].tolist())

    # 4. Extract Dev Features & Tune
    dev_features_df = extract_features_and_score(pipeline, train_dev_df)
    scorer, tau_star = tune_dev_split(dev_features_df)

    # Update pipeline with tuned scorer & threshold
    pipeline.scorer = scorer
    pipeline.scorer.threshold = tau_star

    # 5. Evaluate ONCE on Held-out Test Split
    print("\n--- Evaluating Held-Out Test Split (70%, N = {}) ---".format(len(test_df)))
    test_features_df = extract_features_and_score(pipeline, test_df)

    test_scores = (
        scorer.w_spread * test_features_df["spread"]
        + scorer.w_polysemy * test_features_df["baseline_polysemy"]
        + scorer.w_diversity * test_features_df["domain_diversity"]
    ).values

    y_test_true = test_features_df["gold_is_ambig"].values
    y_test_pred = test_scores >= tau_star

    # Predicted Types
    pred_types = []
    for i, row in test_features_df.iterrows():
        if not y_test_pred[i]:
            pred_types.append("None")
        else:
            sigma = row.get("sigma_star", [])
            pred_types.append(classify_ambiguity(sigma) if sigma else "Intra-Domain")

    gold_types = test_features_df["gold_label"].values

    # Headline Test Metrics
    m = prf1(y_test_true, y_test_pred)
    type_correct = sum(
        1 for t, p in zip(gold_types, pred_types) if t != "None" and t == p
    )
    type_total = sum(1 for t in gold_types if t != "None")
    type_acc = type_correct / type_total if type_total else 0.0

    print("\nHEADLINE TEST RESULTS (Held-Out Test Set):")
    print(f"  Accuracy:  {m['accuracy'] * 100:.2f}%")
    print(f"  Precision: {m['precision'] * 100:.2f}%")
    print(f"  Recall:    {m['recall'] * 100:.2f}%")
    print(f"  F1-Score:  {m['f1'] * 100:.2f}%")
    print(f"  TP: {m['tp']}, FP: {m['fp']}, FN: {m['fn']}, TN: {m['tn']}")
    print(f"\nInter- vs Intra-Domain Classification Accuracy (on truly ambiguous gold items):")
    print(f"  {type_correct} / {type_total} = {type_acc * 100:.2f}%")

    # 6. Non-Parametric Bootstrapping (1,000 resamples for 95% CIs)
    print("\nComputing 95% Confidence Intervals via Bootstrapping (B = 1,000)...")
    ci = bootstrap_ci(y_test_true, y_test_pred, gold_types, pred_types, n_bootstraps=1000, seed=42)
    print(f"  Accuracy 95% CI:   [{ci['accuracy_ci'][0]*100:.1f}%, {ci['accuracy_ci'][1]*100:.1f}%]")
    print(f"  Precision 95% CI:  [{ci['precision_ci'][0]*100:.1f}%, {ci['precision_ci'][1]*100:.1f}%]")
    print(f"  Recall 95% CI:     [{ci['recall_ci'][0]*100:.1f}%, {ci['recall_ci'][1]*100:.1f}%]")
    print(f"  F1-Score 95% CI:   [{ci['f1_ci'][0]*100:.1f}%, {ci['f1_ci'][1]*100:.1f}%]")
    print(f"  Type Acc 95% CI:   [{ci['type_acc_ci'][0]*100:.1f}%, {ci['type_acc_ci'][1]*100:.1f}%]")

    # 7. Precision-Recall Curve Generation & Plot
    precisions_curve, recalls_curve, thresholds_curve = precision_recall_curve(y_test_true, test_scores)
    pr_auc = average_precision_score(y_test_true, test_scores)
    print(f"\nPrecision-Recall AUC (PR-AUC): {pr_auc:.3f}")

    os.makedirs("sample_output", exist_ok=True)
    plt.figure(figsize=(7, 5))
    plt.plot(recalls_curve, precisions_curve, color="#2BC4B8", lw=2.5, label=f"Proposed Framework (PR-AUC = {pr_auc:.2f})")
    plt.scatter([m["recall"]], [m["precision"]], color="#E8935A", s=90, zorder=5, label=f"Operating Point (tau*={tau_star:.2f})")
    plt.xlabel("Recall", fontsize=11)
    plt.ylabel("Precision", fontsize=11)
    plt.title("Precision-Recall Curve on Held-Out Test Split (N = 175)", fontsize=12)
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("sample_output/precision_recall_curve.png", dpi=200)
    plt.close()

    # Save PR curve data
    df_pr = pd.DataFrame({
        "precision": precisions_curve,
        "recall": recalls_curve,
        "threshold": list(thresholds_curve) + [1.0]
    })
    df_pr.to_csv("sample_output/precision_recall_curve_data.csv", index=False)

    # 8. Save Detailed Test Predictions & Summary
    test_results_df = test_features_df.copy()
    test_results_df["predicted_is_ambig"] = y_test_pred
    test_results_df["predicted_label"] = pred_types
    test_results_df["is_correct"] = (test_results_df["gold_label"] == test_results_df["predicted_label"])
    test_results_df = test_results_df.drop(columns=["sigma_star", "ranked_senses"], errors="ignore")
    test_results_df.to_csv("sample_output/test_evaluation_results.csv", index=False)

    summary_text = f"""HELD-OUT TEST SET EVALUATION SUMMARY (Review Item 2)
============================================================
Benchmark Split: 30% Dev (N = {len(train_dev_df)}) / 70% Test (N = {len(test_df)})
Hyperparameters Tuned on Dev:
  - Operating Threshold tau*: {tau_star:.2f}
  - Learned Feature Weights:
      w_spread:     {scorer.w_spread:.3f}
      w_polysemy:   {scorer.w_polysemy:.3f}
      w_diversity:  {scorer.w_diversity:.3f}

Headline Metrics on Held-Out Test Set (Evaluated ONCE):
  - Accuracy:   {m['accuracy'] * 100:.2f}%  (95% CI: [{ci['accuracy_ci'][0]*100:.1f}%, {ci['accuracy_ci'][1]*100:.1f}%])
  - Precision:  {m['precision'] * 100:.2f}%  (95% CI: [{ci['precision_ci'][0]*100:.1f}%, {ci['precision_ci'][1]*100:.1f}%])
  - Recall:     {m['recall'] * 100:.2f}%  (95% CI: [{ci['recall_ci'][0]*100:.1f}%, {ci['recall_ci'][1]*100:.1f}%])
  - F1-Score:   {m['f1'] * 100:.2f}%  (95% CI: [{ci['f1_ci'][0]*100:.1f}%, {ci['f1_ci'][1]*100:.1f}%])
  - PR-AUC:     {pr_auc:.3f}

Confusion Matrix (Detection):
  TP: {m['tp']}  FP: {m['fp']}  FN: {m['fn']}  TN: {m['tn']}

Inter- vs Intra-Domain Classification (Truly Ambiguous Cases):
  - Accuracy:   {type_acc * 100:.2f}% ({type_correct}/{type_total})  (95% CI: [{ci['type_acc_ci'][0]*100:.1f}%, {ci['type_acc_ci'][1]*100:.1f}%])

Classification Report (All 3 Classes):
{classification_report(gold_types, pred_types, zero_division=0)}
"""
    with open("sample_output/test_evaluation_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)

    print("\nWrote sample_output/test_evaluation_results.csv, test_evaluation_summary.txt, and precision_recall_curve.png")
    return m, ci, pr_auc


if __name__ == "__main__":
    evaluate_pipeline()
