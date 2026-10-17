"""
STEP 1 - Prepare a clean, leak-free, mixed-length dataset.

Input  (put ONE of these options in ./data/):
    a) data/Fake.csv + data/True.csv       (Kaggle ISOT "Fake and Real News")
    b) ONE combined csv with fake + real news, e.g. data/news.csv
       -> columns and labels are auto-detected; the script prints examples of each
          label so you can confirm which value means FAKE.
          python prepare_data.py --combined data/news.csv [--fake_value 0]
    data/extra/*.csv   (OPTIONAL, e.g. Indian news - columns: text,label[,title]
                        label may be 0/1 or FAKE/REAL)

Output:
    data/processed/train.csv, val.csv, test.csv      (columns: text,label,kind)
    data/processed/test_articles.csv                 (full test articles)
    reports/leak_check.txt

Run:  python prepare_data.py
"""
import argparse
import glob
import os
import random

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from text_utils import clean_text, is_tweet_dump, make_samples

SEED = 42
DATA_DIR = "data"
OUT_DIR = os.path.join(DATA_DIR, "processed")
REPORT_DIR = "reports"
MIN_WORDS = 20
LABEL_NAMES = {0: "FAKE", 1: "REAL"}


def load_isot():
    fake = pd.read_csv(os.path.join(DATA_DIR, "Fake.csv"))
    true = pd.read_csv(os.path.join(DATA_DIR, "True.csv"))
    fake["label"], true["label"] = 0, 1
    fake["source"], true["source"] = "isot", "isot"
    return pd.concat([fake, true], ignore_index=True)[["title", "text", "label", "source"]]


# ---------------------------------------------------------------------------
# ONE combined CSV (fake + real in the same file)
# ---------------------------------------------------------------------------
TEXT_COLS = ["text", "content", "article", "news", "body", "clean_content", "clean_text", "statement"]
TITLE_COLS = ["title", "headline", "heading"]
LABEL_COLS = ["label", "labels", "class", "target", "is_fake", "fake", "category", "y"]
WORD_LABELS = {"FAKE": 0, "FALSE": 0, "F": 0, "UNRELIABLE": 0,
               "REAL": 1, "TRUE": 1, "T": 1, "RELIABLE": 1}


def _pick(cols, wanted, given, what):
    if given:
        if given not in cols:
            raise SystemExit(f"Column '{given}' not found. Columns in your file: {list(cols)}")
        return given
    lower = {c.lower().strip(): c for c in cols}
    for w in wanted:
        if w in lower:
            return lower[w]
    if what == "title":
        return None
    raise SystemExit(f"Could not find the {what} column. Columns in your file: {list(cols)}\n"
                     f"Re-run with --{what}_col <column name>")


def load_combined(path, text_col=None, title_col=None, label_col=None, fake_value=None):
    print(f"Reading combined file: {path}")
    d = pd.read_csv(path)
    print(f"  columns: {list(d.columns)} | rows: {len(d)}")
    tc = _pick(d.columns, TEXT_COLS, text_col, "text")
    hc = _pick(d.columns, TITLE_COLS, title_col, "title")
    lc = _pick(d.columns, LABEL_COLS, label_col, "label")
    print(f"  using text='{tc}', title='{hc}', label='{lc}'")

    d = d.dropna(subset=[tc, lc])
    raw = d[lc].astype(str).str.strip().str.upper().str.replace(r"\.0$", "", regex=True)
    values = sorted(raw.unique())
    print(f"  label values found: {values}  counts: {raw.value_counts().to_dict()}")
    if len(values) != 2:
        raise SystemExit("The label column must have exactly 2 values (fake / real).")

    # --- Show evidence so YOU can confirm which value means FAKE -------------
    reuters_share = {}
    print("\n  ---- LABEL CHECK: 2 examples per label value ----")
    for v in values:
        texts = d.loc[raw == v, tc].astype(str)
        reuters_share[v] = texts.str.contains(r"\breuters\b", case=False, regex=True).mean()
        print(f"\n  label = {v}   ({reuters_share[v]:.0%} of rows mention 'Reuters')")
        for t in texts.sample(n=min(2, len(texts)), random_state=1):
            print("    >", t[:220].replace("\n", " "), "...")
    print("  ------------------------------------------------\n")

    if all(v in WORD_LABELS for v in values):
        mapping = {v: WORD_LABELS[v] for v in values}
        how = "from the label words"
    elif fake_value is not None:
        fv = str(fake_value).strip().upper()
        if fv not in values:
            raise SystemExit(f"--fake_value {fake_value} is not one of {values}")
        mapping = {v: (0 if v == fv else 1) for v in values}
        how = "from --fake_value"
    else:
        hi, lo = max(values, key=reuters_share.get), min(values, key=reuters_share.get)
        if reuters_share[hi] > 0.5 and reuters_share[lo] < 0.05:
            mapping = {hi: 1, lo: 0}
            how = f"inferred: label {hi} is mostly Reuters articles, so it is REAL"
        else:
            raise SystemExit(
                "Cannot tell which label value means FAKE.\n"
                "Read the examples above, then re-run, e.g.:\n"
                f"    python prepare_data.py --combined {path} --fake_value {values[0]}")
    print("  LABEL MAPPING:", {k: LABEL_NAMES[v] for k, v in mapping.items()}, f"({how})")
    if len(set(mapping.values())) != 2:
        raise SystemExit("Both label values map to the same class - check the label column.")

    has_dots = d[tc].astype(str).str.contains(r"[.!?]", regex=True).mean()
    if has_dots < 0.3:
        print("  NOTE: this text looks pre-processed (little punctuation). BERT works best on the\n"
              "        ORIGINAL text - if you have the raw version of this file, use that instead.")

    out = pd.DataFrame({
        "title": d[hc].fillna("").astype(str) if hc else "",
        "text": d[tc].astype(str),
        "label": raw.map(mapping).astype(int),
        "source": os.path.basename(path),
    })
    return out


def load_extra():
    frames = []
    for path in glob.glob(os.path.join(DATA_DIR, "extra", "*.csv")):
        d = pd.read_csv(path)
        d.columns = [c.lower().strip() for c in d.columns]
        if "text" not in d.columns or "label" not in d.columns:
            print(f"  ! skipped {path}: needs 'text' and 'label' columns")
            continue
        lab = d["label"].astype(str).str.upper().str.strip()
        d["label"] = lab.map({"0": 0, "1": 1, "FAKE": 0, "REAL": 1, "FALSE": 0, "TRUE": 1})
        d = d.dropna(subset=["label"])
        d["label"] = d["label"].astype(int)
        if "title" not in d.columns:
            d["title"] = ""
        d["source"] = os.path.basename(path)
        frames.append(d[["title", "text", "label", "source"]])
        print(f"  + extra dataset {path}: {len(d)} rows")
    return pd.concat(frames, ignore_index=True) if frames else None


def leak_check(train_df, test_df):
    """Train a quick TF-IDF + LogisticRegression and list its strongest words.
    If words like 'reuters', 'featured', 'image', 'via', 'wire', 'getty'
    appear at the top, the cleaning missed a fingerprint."""
    vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), sublinear_tf=True, min_df=3)
    xtr = vec.fit_transform(train_df["text"])
    xte = vec.transform(test_df["text"])
    clf = LogisticRegression(max_iter=2000, C=4.0).fit(xtr, train_df["label"])
    acc = accuracy_score(test_df["label"], clf.predict(xte))
    feats = vec.get_feature_names_out()
    order = clf.coef_[0].argsort()
    top_fake = [feats[i] for i in order[:30]]
    top_real = [feats[i] for i in order[-30:][::-1]]
    suspicious = {"reuters", "featured", "image", "via", "wire", "getty", "century", "pic", "twitter com"}
    found = sorted(suspicious & set(top_fake + top_real))

    lines = [
        f"TF-IDF + LogisticRegression accuracy on full test articles: {acc:.4f}",
        "",
        "Top words that push towards FAKE:", ", ".join(top_fake),
        "",
        "Top words that push towards REAL:", ", ".join(top_real),
        "",
        ("WARNING: possible leak words still present: " + ", ".join(found)) if found
        else "OK: no known source fingerprints among the top features.",
    ]
    report = "\n".join(lines)
    with open(os.path.join(REPORT_DIR, "leak_check.txt"), "w", encoding="utf-8") as f:
        f.write(report)
    print(report)


def expand(df, seed):
    rng = random.Random(seed)
    rows = []
    for t, x, y in zip(df["title"], df["text"], df["label"]):
        rows.extend(make_samples(t, x, int(y), rng))
    out = pd.DataFrame(rows, columns=["text", "label", "kind"])
    return out.drop_duplicates(subset=["text"]).sample(frac=1, random_state=seed).reset_index(drop=True)


def parse_args():
    p = argparse.ArgumentParser(description="Prepare clean, leak-free training data")
    p.add_argument("--combined", help="ONE csv containing both fake and real news, e.g. data/news.csv")
    p.add_argument("--text_col", help="name of the article text column (auto-detected if omitted)")
    p.add_argument("--title_col", help="name of the title column (optional)")
    p.add_argument("--label_col", help="name of the label column (auto-detected if omitted)")
    p.add_argument("--fake_value", help="the label value that means FAKE, e.g. 0 or 1")
    return p.parse_args()


def find_input(args):
    if args.combined:
        return "combined", args.combined
    if os.path.exists(os.path.join(DATA_DIR, "Fake.csv")) and os.path.exists(os.path.join(DATA_DIR, "True.csv")):
        return "isot", None
    csvs = [p for p in glob.glob(os.path.join(DATA_DIR, "*.csv"))]
    if len(csvs) == 1:
        return "combined", csvs[0]
    raise SystemExit("Put either Fake.csv + True.csv, or ONE combined csv, into the data/ folder\n"
                     "(or pass --combined path/to/file.csv). Found: " + str(csvs))


def main():
    args = parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    print("Loading data ...")
    kind, path = find_input(args)
    if kind == "isot":
        df = load_isot()
    else:
        df = load_combined(path, args.text_col, args.title_col, args.label_col, args.fake_value)
    extra = load_extra()
    if extra is not None:
        df = pd.concat([df, extra], ignore_index=True)
    print(f"  raw rows: {len(df)}")

    df["title"] = df["title"].fillna("").astype(str)
    df["text"] = df["text"].fillna("").astype(str)
    df = df[~df["text"].apply(is_tweet_dump)]

    print("Removing source fingerprints (Reuters dateline, image credits, links ...)")
    df["title"] = df["title"].apply(clean_text)
    df["text"] = df["text"].apply(clean_text)

    df = df[df["text"].str.split().str.len() >= MIN_WORDS]
    df = df.drop_duplicates(subset=["text"])
    has_title = df["title"] != ""
    df = pd.concat([df[has_title].drop_duplicates(subset=["title"]), df[~has_title]])
    df = df.reset_index(drop=True)
    print(f"  clean unique articles: {len(df)}")
    print("  label balance:", df["label"].map(LABEL_NAMES).value_counts().to_dict())

    # Split by ARTICLE first (70 / 15 / 15) - before making short/medium versions,
    # so no piece of a test article is ever seen in training (no leakage).
    train_a, temp_a = train_test_split(df, test_size=0.30, random_state=SEED, stratify=df["label"])
    val_a, test_a = train_test_split(temp_a, test_size=0.50, random_state=SEED, stratify=temp_a["label"])
    print(f"  articles -> train {len(train_a)}, val {len(val_a)}, test {len(test_a)}")

    print("\nLeak check (quick baseline):")
    leak_check(train_a, test_a)

    train = expand(train_a, SEED)
    val = expand(val_a, SEED + 1)
    test = expand(test_a, SEED + 2)

    train.to_csv(os.path.join(OUT_DIR, "train.csv"), index=False)
    val.to_csv(os.path.join(OUT_DIR, "val.csv"), index=False)
    test.to_csv(os.path.join(OUT_DIR, "test.csv"), index=False)
    test_a[["title", "text", "label"]].to_csv(os.path.join(OUT_DIR, "test_articles.csv"), index=False)

    print("\nSamples per length type (train):")
    print(train["kind"].value_counts().to_string())
    print(f"\nSaved: train {len(train)}, val {len(val)}, test {len(test)} -> {OUT_DIR}/")
    print("Next: python train_bert.py")


if __name__ == "__main__":
    main()
