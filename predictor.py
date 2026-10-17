"""
Prediction logic shared by the Flask app and the training script.

* Cleans input with the SAME clean_text() used in training.
* Reads the label names from the model config (no hard-coded 0/1 guessing).
* Long articles are split into overlapping 256-token chunks; the chunk
  probabilities are averaged, so the whole article is used, not just the start.

Quick test:  python predictor.py "Some news text here"
"""
import json
import os
import sys

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from text_utils import clean_text

UNCERTAIN_BELOW = 0.65     # if the top probability is lower, show "UNCERTAIN"
SHORT_INPUT_WORDS = 8      # below this, show a "too short" warning


class FakeNewsPredictor:
    def __init__(self, model_dir="model", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device)
        self.model.eval()
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        cfg_path = os.path.join(model_dir, "inference_config.json")
        self.max_len = 256
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                self.max_len = json.load(f).get("max_len", 256)
        if set(self.id2label.values()) != {"FAKE", "REAL"}:
            raise ValueError(f"Unexpected labels in model config: {self.id2label}. "
                             "Retrain with train_bert.py so labels are saved as FAKE/REAL.")

    def _chunks(self, ids):
        body = self.max_len - 2                     # room for [CLS] and [SEP]
        if len(ids) <= body:
            return [ids]
        stride = body // 2                          # 50% overlap
        chunks = [ids[i:i + body] for i in range(0, len(ids) - stride, stride)]
        return chunks[:16]                          # cap for very long texts

    @torch.no_grad()
    def predict(self, text):
        cleaned = clean_text(text)
        words = len(cleaned.split())
        if words == 0:
            raise ValueError("Empty input")

        ids = self.tokenizer(cleaned, add_special_tokens=False, verbose=False)["input_ids"]
        chunks = self._chunks(ids)
        cls, sep = self.tokenizer.cls_token_id, self.tokenizer.sep_token_id
        pad = self.tokenizer.pad_token_id or 0
        batch = [[cls] + c + [sep] for c in chunks]
        width = max(len(b) for b in batch)
        input_ids = torch.tensor([b + [pad] * (width - len(b)) for b in batch], device=self.device)
        attention = torch.tensor([[1] * len(b) + [0] * (width - len(b)) for b in batch], device=self.device)
        logits = self.model(input_ids=input_ids, attention_mask=attention).logits
        probs = torch.softmax(logits, dim=-1).mean(dim=0).cpu().tolist()

        by_label = {self.id2label[i]: p for i, p in enumerate(probs)}
        label_raw = max(by_label, key=by_label.get)
        confidence = by_label[label_raw]

        warning = None
        if words < SHORT_INPUT_WORDS:
            warning = ("Very short input. The model judges the writing style of news "
                       "articles; it cannot verify single facts. Paste a full paragraph "
                       "for a meaningful result.")

        return {
            "label": label_raw if confidence >= UNCERTAIN_BELOW else "UNCERTAIN",
            "label_raw": label_raw,
            "confidence": confidence,
            "prob_fake": by_label["FAKE"],
            "prob_real": by_label["REAL"],
            "chunks": len(chunks),
            "words": words,
            "warning": warning,
        }


if __name__ == "__main__":
    text = " ".join(sys.argv[1:]) or "The government announced a new education policy on Tuesday."
    print(json.dumps(FakeNewsPredictor("model").predict(text), indent=2))
