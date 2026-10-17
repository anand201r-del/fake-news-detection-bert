"""
STEP 2 - Fine-tune BERT on the clean mixed-length data.

Recommended: run on Google Colab with a GPU (see colab_train.ipynb).
On a laptop CPU, use a small test run first:
    python train_bert.py --max_train 2000 --epochs 1

Full run (Colab GPU, ~1-1.5 h on a T4):
    python train_bert.py

Output:
    model/                      <- the folder the Flask app loads
    reports/metrics.json        <- numbers for your thesis / resume
    reports/confusion_matrix.png
"""
import argparse
import inspect
import json
import os

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)

from predictor import FakeNewsPredictor

ID2LABEL = {0: "FAKE", 1: "REAL"}
LABEL2ID = {"FAKE": 0, "REAL": 1}
DATA = os.path.join("data", "processed")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", default="bert-base-uncased")
    p.add_argument("--out_dir", default="model")
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max_train", type=int, default=0, help="limit training samples (0 = all)")
    p.add_argument("--max_eval", type=int, default=5000, help="limit validation samples")
    return p.parse_args()


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
    }


def make_training_args(a):
    kwargs = dict(
        output_dir="checkpoints",
        num_train_epochs=a.epochs,
        per_device_train_batch_size=a.batch,
        per_device_eval_batch_size=a.batch * 2,
        learning_rate=a.lr,
        weight_decay=0.01,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=1,
        logging_steps=100,
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=42,
    )
    # the argument was renamed in newer transformers versions
    params = inspect.signature(TrainingArguments.__init__).parameters
    kwargs["eval_strategy" if "eval_strategy" in params else "evaluation_strategy"] = "epoch"
    kwargs["warmup_steps"] = 200
    unknown = [k for k in kwargs if k not in params]
    for k in unknown:
        kwargs.pop(k)
    if unknown:
        print("Skipping arguments not supported by this transformers version:", unknown)
    return TrainingArguments(**kwargs)


def main():
    a = parse_args()
    os.makedirs("reports", exist_ok=True)
    print("GPU available:", torch.cuda.is_available())

    train = pd.read_csv(os.path.join(DATA, "train.csv")).dropna()
    val = pd.read_csv(os.path.join(DATA, "val.csv")).dropna()
    test = pd.read_csv(os.path.join(DATA, "test.csv")).dropna()
    if a.max_train:
        train = train.sample(n=min(a.max_train, len(train)), random_state=42)
    if a.max_eval:
        val = val.sample(n=min(a.max_eval, len(val)), random_state=42)
    print(f"train {len(train)} | val {len(val)} | test {len(test)}")

    tok = AutoTokenizer.from_pretrained(a.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        a.base_model, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID)

    def tokenize(batch):
        return tok(batch["text"], truncation=True, max_length=a.max_len)

    def to_ds(df):
        ds = Dataset.from_pandas(df[["text", "label"]].reset_index(drop=True))
        return ds.map(tokenize, batched=True, remove_columns=["text"])

    trainer = Trainer(
        model=model,
        args=make_training_args(a),
        train_dataset=to_ds(train),
        eval_dataset=to_ds(val),
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=compute_metrics,
    )
    trainer.train()

    trainer.save_model(a.out_dir)
    tok.save_pretrained(a.out_dir)
    with open(os.path.join(a.out_dir, "inference_config.json"), "w") as f:
        json.dump({"max_len": a.max_len, "labels": ID2LABEL}, f, indent=2)
    print(f"\nModel saved to ./{a.out_dir}/")

    # ------------------------------------------------------------------
    # Evaluation on the untouched test set, per input length
    # ------------------------------------------------------------------
    print("\nEvaluating on the test set ...")
    pred_out = trainer.predict(to_ds(test))
    preds = pred_out.predictions.argmax(-1)
    test = test.assign(pred=preds)

    results = {}
    for kind, g in [("all", test)] + list(test.groupby("kind")):
        results[kind] = {
            "n": int(len(g)),
            "accuracy": round(accuracy_score(g["label"], g["pred"]), 4),
            "f1": round(f1_score(g["label"], g["pred"]), 4),
            "precision": round(precision_score(g["label"], g["pred"]), 4),
            "recall": round(recall_score(g["label"], g["pred"]), 4),
        }

    # full articles with chunking (how the website works)
    predictor = FakeNewsPredictor(a.out_dir)
    arts = pd.read_csv(os.path.join(DATA, "test_articles.csv")).dropna(subset=["text"])
    arts = arts.sample(n=min(1500, len(arts)), random_state=42)
    chunk_preds = [LABEL2ID[predictor.predict(t)["label_raw"]] for t in arts["text"]]
    results["full_article_chunked"] = {
        "n": int(len(arts)),
        "accuracy": round(accuracy_score(arts["label"], chunk_preds), 4),
        "f1": round(f1_score(arts["label"], chunk_preds), 4),
    }

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
    except Exception as e:  # plotting is optional
        print("Skipped confusion-matrix plot:", e)

    print("\nSanity check on hand-written inputs (style-based, NOT fact checking):")
    for s in ["Narendra Modi is the Prime Minister of India",
              "The government announced on Tuesday that a new education policy will provide "
              "financial assistance to students, officials said.",
              "SHOCKING!!! Hillary caught in secret satanic ritual, the media won't tell you!"]:
        r = predictor.predict(s)
        print(f"  {r['label']:>9}  {r['confidence']:.1%}  | {s[:70]}")

    print("\nDone. Copy the ./model folder next to app.py and run: python app.py")


if __name__ == "__main__":
    main()
