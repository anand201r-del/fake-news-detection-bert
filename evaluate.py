"""
Evaluate an already-trained model (no retraining).

Run:  python evaluate.py            (uses ./model and data/processed/)
Output: reports/metrics.json, reports/confusion_matrix.png
"""
import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)

from predictor import FakeNewsPredictor

DATA = os.path.join("data", "processed")
MODEL_DIR = "model"
LABEL2ID = {"FAKE": 0, "REAL": 1}


@torch.no_grad()
def batch_predict(pred, texts, batch_size=64):
    out = []
    for i in range(0, len(texts), batch_size):
        enc = pred.tokenizer(list(texts[i:i + batch_size]), truncation=True, max_length=pred.max_len,
                             padding=True, return_tensors="pt").to(pred.device)
        logits = pred.model(**enc).logits
        ids = logits.argmax(-1).cpu().tolist()
        out.extend(LABEL2ID[pred.id2label[j]] for j in ids)
        if (i // batch_size) % 50 == 0:
            print(f"  {i + len(ids)}/{len(texts)}")
    return out


def scores(y, p):
    return {"n": int(len(y)),
            "accuracy": round(accuracy_score(y, p), 4),
            "f1": round(f1_score(y, p), 4),
            "precision": round(precision_score(y, p), 4),
            "recall": round(recall_score(y, p), 4)}


def main():
    os.makedirs("reports", exist_ok=True)
    pred = FakeNewsPredictor(MODEL_DIR)
    print("Labels:", pred.id2label, "| device:", pred.device)

    test = pd.read_csv(os.path.join(DATA, "test.csv")).dropna()
    print(f"Predicting {len(test)} test samples ...")
    test["pred"] = batch_predict(pred, test["text"].astype(str).tolist())

    results = {"all": scores(test["label"], test["pred"])}
    for kind, g in test.groupby("kind"):
        results[kind] = scores(g["label"], g["pred"])

    print("Full articles with chunking ...")
    arts = pd.read_csv(os.path.join(DATA, "test_articles.csv")).dropna(subset=["text"])
    arts = arts.sample(n=min(1500, len(arts)), random_state=42)
    chunk_preds = [LABEL2ID[pred.predict(t)["label_raw"]] for t in arts["text"]]
    results["full_article_chunked"] = scores(arts["label"], chunk_preds)

    print(json.dumps(results, indent=2))
    print(classification_report(test["label"], test["pred"], target_names=["FAKE", "REAL"], digits=4))
    with open(os.path.join("reports", "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cm = confusion_matrix(test["label"], test["pred"])
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha="center", va="center",
                    color="white" if v > cm.max() / 2 else "black")
        ax.set_xticks([0, 1], ["FAKE", "REAL"])
        ax.set_yticks([0, 1], ["FAKE", "REAL"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("BERT - test set")
        fig.tight_layout()
        fig.savefig(os.path.join("reports", "confusion_matrix.png"), dpi=150)
    except Exception as e:
        print("Skipped plot:", e)

    print("\nSanity check on hand-written inputs (style-based, NOT fact checking):")
    for s in ["Narendra Modi is the Prime Minister of India",
              "The government announced on Tuesday that a new education policy will provide "
              "financial assistance to students, officials said.",
              "SHOCKING!!! Hillary caught in secret satanic ritual, the media won't tell you!"]:
        r = pred.predict(s)
        print(f"  {r['label']:>9}  {r['confidence']:.1%}  | {s[:70]}")
    print("\nSaved reports/metrics.json and reports/confusion_matrix.png")


if __name__ == "__main__":
    main()
