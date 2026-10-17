"""
Shared text utilities.

IMPORTANT: prepare_data.py, train_bert.py, predictor.py and app.py ALL import
clean_text() from this file, so training and the website always see text that
was cleaned in exactly the same way.
"""
import random
import re

# ---------------------------------------------------------------------------
# 1. Leak removal
# ---------------------------------------------------------------------------
# The Kaggle ISOT dataset (Fake.csv / True.csv) contains "source fingerprints"
# that let a model cheat:
#   * almost every REAL article starts with "WASHINGTON (Reuters) - "
#   * many FAKE articles contain "Featured image via ...", "21st Century Wire",
#     "(VIDEO)", twitter links, @handles, "Getty Images", ...
# A model trained on the raw data learns "has Reuters => REAL, otherwise FAKE",
# which is why the old model scored 99.99% but called every typed sentence FAKE.

_PATTERNS = [
    # "WASHINGTON (Reuters) - ", "LONDON/PARIS (Reuters) - " at the start
    (r"^\s*[^()]{0,120}?\(\s*reuters\s*\)\s*[-–—:]*\s*", ""),
    (r"\(\s*reuters\s*\)", " "),
    (r"\breuters\b", " "),
    # footers / credits typical of the fake-news sites
    (r"featured image (via|by|credit)[^\n]*$", " "),
    (r"\b(image|photo|screenshot|screen capture|photograph)s? (via|by|credit|courtesy of)\b[^.\n]*", " "),
    (r"\bgetty images?\b", " "),
    (r"\b21st century wire\b( says)?", " "),
    (r"\bread more:?[^\n]*$", " "),
    (r"\bsupport (us|our work)\b[^\n]*$", " "),
    (r"\bvia\b", " "),
    (r"\bh/t\b", " "),
    # links, handles, twitter embeds
    (r"pic\.twitter\.com/\S+", " "),
    (r"https?://\S+|www\.\S+|\b\S+\.(com|org|net|co)/\S*", " "),
    (r"@\w+", " "),
    (r"#(\w+)", r"\1"),
    (r"[—\-]\s*[A-Z][\w .]+\(@?\w+\)\s+\w+ \d{1,2}, \d{4}", " "),  # tweet signature lines
    # bracketed media tags: (VIDEO) [WATCH] (TWEETS) ...
    (r"[\[(]\s*(video|videos|watch|tweet|tweets|image|images|photo|photos|details|breaking|audio|graphic)\s*[\])]", " "),
    (r"^\s*(breaking|watch|just in|wow|boom)\s*[:!]+", " "),
]
_COMPILED = [(re.compile(p, re.IGNORECASE | re.MULTILINE), r) for p, r in _PATTERNS]

# Same dateline in already lower-cased, punctuation-free text:
# "washington reuters the head of ..." (case-sensitive: lowercase text only)
_LOWER_DATELINE = re.compile(r"^\s*(?:[a-z,/.'&-]+\s+){1,4}reuters\b\s*[-\u2013\u2014:]*\s*")


def clean_text(text) -> str:
    """Remove dataset source fingerprints and normalise whitespace.

    Casing is kept here (the uncased BERT tokenizer lower-cases anyway).
    """
    if text is None:
        return ""
    text = str(text)
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    text = _LOWER_DATELINE.sub("", text)
    for pat, repl in _COMPILED:
        text = pat.sub(repl, text)
    text = re.sub(r"\s+", " ", text).strip(" -–—:")
    return text.strip()


# Real-news rows that are not news articles (raw dumps of Trump's tweets).
TWEET_DUMP_MARKERS = (
    "the following statements were posted to the verified twitter accounts",
    "the following statements were posted to the verified twitter account",
)


def is_tweet_dump(text: str) -> bool:
    t = str(text).lower()[:300]
    return any(m in t for m in TWEET_DUMP_MARKERS)


# ---------------------------------------------------------------------------
# 2. Mixed-length samples (short sentence -> full article)
# ---------------------------------------------------------------------------
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")


def split_sentences(text: str):
    sents = [s.strip() for s in _SENT_SPLIT.split(text) if len(s.split()) >= 3]
    words = text.split()
    if len(sents) <= 1 and len(words) > 40:
        # text without punctuation (already pre-processed): use 20-word pieces
        sents = [" ".join(words[i:i + 20]) for i in range(0, len(words), 20)]
    return sents


def make_samples(title: str, text: str, label: int, rng: random.Random):
    """Create several training examples of different lengths from ONE article.

    kinds:
      title  - the headline (short)
      short  - 1-2 consecutive sentences from a RANDOM position (<= 40 words)
      medium - 3-6 consecutive sentences from a RANDOM position (<= 150 words)
      full   - the whole article (long ones are truncated / chunked later)

    Windows start at random positions, not only at the beginning, so the model
    cannot depend on how an article opens.
    """
    out = []
    title = (title or "").strip()
    if len(title.split()) >= 3:
        out.append((title, label, "title"))

    sents = split_sentences(text)
    if sents:
        n = rng.randint(1, 2)
        i = rng.randrange(0, max(1, len(sents) - n + 1))
        short = " ".join(sents[i:i + n])
        short = " ".join(short.split()[:40])
        if len(short.split()) >= 5:
            out.append((short, label, "short"))

    if len(sents) >= 4:
        n = rng.randint(3, min(6, len(sents)))
        i = rng.randrange(0, len(sents) - n + 1)
        medium = " ".join(" ".join(sents[i:i + n]).split()[:150])
        out.append((medium, label, "medium"))

    out.append((text, label, "full"))
    return out
