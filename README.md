# TruthLens: Fake News Detection using BERT

A fake news classifier built on a fine-tuned **BERT** model, with a multi-page **Flask** web app.
It handles everything from a single headline to a long article and reports a confidence score.

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c) ![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow) ![Flask](https://img.shields.io/badge/Flask-web%20app-black)

## Results (held-out test set)

| Input type | Samples | Accuracy | F1 |
|---|---:|---:|---:|
| Headline | 5,685 | 97.17% | 97.44% |
| Short (1–2 sentences) | 5,611 | 95.03% | 95.58% |
| Medium (3–6 sentences) | 5,061 | 98.74% | 98.81% |
| Full article (first 256 tokens) | 5,502 | 99.65% | 99.68% |
| Full article, chunked (used by the app) | 1,500 | 99.53% | 99.59% |
| **All** | **21,859** | **97.61%** | **97.82%** |

Precision 96.98% · Recall 98.68%

## Removing a data leak

The first version of this project scored **99.99%** but labelled almost every typed sentence as FAKE.
The reason was a **data leak** in the ISOT dataset: nearly every real article starts with
`WASHINGTON (Reuters) -`, and fake articles carry tags like `Featured image via…`, `(VIDEO)` or twitter links.
The model had learned *“Reuters = real”* instead of learning the language itself.

This version fixes that:

1. **Leak removal:** Reuters datelines, image credits, links, @handles and “via” are stripped (`text_utils.clean_text`), with the same cleaning used in training and in the app.
2. **Leak check:** a TF-IDF + Logistic Regression baseline lists its most important words, to confirm no source names remain.
3. **Split before augmentation:** articles are split 70/15/15 *first*, so no piece of a test article is ever seen in training.
4. **Mixed-length training:** each article becomes a headline, 1–2 sentences, 3–6 sentences (taken from random positions) and the full text.
5. **Chunked inference:** long articles are split into overlapping 256-token windows and the probabilities are averaged.
6. **Honest output:** confidence below 65% is shown as *Uncertain*, and very short inputs get a warning.

## Web app

| Page | Description |
|---|---|
| **Analyze** | Paste text and get the verdict, a confidence gauge and FAKE/REAL probabilities (no page reload) |
| **History** | Past checks with filters, search and summary counts (stored in the browser) |
| **Performance** | Test metrics, accuracy by input length, confusion matrix |
| **How it works** | Pipeline overview, strengths and limitations |
| **About** | Project summary, tech stack, API |

REST API:

```bash
curl -X POST http://127.0.0.1:5000/api/predict -H "Content-Type: application/json" \
     -d "{\"text\": \"The government announced on Tuesday ...\"}"
```

## Project structure

```
├── app.py               Flask app (pages + /api/predict)
├── predictor.py         Loads the model, cleans input, chunked prediction
├── text_utils.py        Leak removal + short/medium/long sample creation
├── prepare_data.py      Step 1: clean, dedupe, split, augment, leak check
├── train_bert.py        Step 2: fine-tune bert-base-uncased
├── evaluate.py          Evaluate a saved model (metrics.json, confusion matrix)
├── baselines.py         Optional: Logistic Regression / Naive Bayes / SVM on the same data
├── colab_train.ipynb    Runs steps 1–2 on a free Colab GPU
├── templates/           HTML pages (base, analyze, history, performance, how, about)
├── reports/             metrics.json, confusion_matrix.png
├── data/                (not in repo) put the dataset here
└── model/               (not in repo) trained model goes here
```

## How to run

### 1. Get the dataset
Download the **ISOT Fake and Real News** dataset ([Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset))
and put `Fake.csv` and `True.csv` in `data/`. A single combined CSV also works: `python prepare_data.py --combined data/news.csv`.

### 2. Train (Google Colab, T4 GPU, about 1.5 h)
Upload the folder to Google Drive, open `colab_train.ipynb`, and choose **Runtime → Change runtime type → T4 GPU**. Then run:

```bash
python prepare_data.py                          # check the leak report says OK
python train_bert.py --max_train 60000 --epochs 2
python evaluate.py                              # writes reports/metrics.json
```

The trained model is about 420 MB, which is too large for GitHub. Download the `model/` folder from Drive
and put it next to `app.py`.

### 3. Run the web app

```bash
python -m venv venv
venv\Scripts\activate          # Windows  (Linux/Mac: source venv/bin/activate)
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000

## Limitations

- The model judges **writing style**, not facts. A false claim written in a calm, neutral way can be classified as real.
- The training data is mostly 2016–17 US political news, so accuracy on other regions and topics (e.g. Indian news) is lower.
- One-line statements carry little style information. Full paragraphs work best.

**Future work:** add a fact-verification module (evidence retrieval + NLI) for short claims, and train on Indian news datasets such as IFND.

## Tech stack
Python · PyTorch · Hugging Face Transformers · BERT · scikit-learn · pandas · Flask · HTML/CSS/JS · Google Colab

## Author
**Anand Kumar**, M.Tech CSE, IIIT Guwahati · [GitHub](https://github.com/anand201r-del)
