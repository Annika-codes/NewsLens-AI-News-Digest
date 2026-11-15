"""
Train NewsLens's tone classifier and export it for the browser.

Pipeline:  labels.csv  ->  TF-IDF (words + word pairs)  ->  logistic regression
           -> honest evaluation (nested cross-validation) -> ../tone_model.json

Usage:
    pip install -r requirements.txt
    python train_tone.py --data labels.csv --out ../tone_model.json

labels.csv comes from label_tool.html and has two columns: text,label
Labels are NEUTRAL, EMOTIONAL, OPINIONATED.
"""
import argparse
import csv
import json
import re
import sys
from collections import Counter
from datetime import date

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

# The tokenizer MUST match tokenize() in index.html: lowercase, runs of 2+ letters/digits.
TOKEN_PATTERN = r"[a-z0-9]{2,}"

# The keyword rules the page used before it had a model. Used as a baseline to beat.
LEX = {
    "EMOTIONAL": ["shock", "devastat", "heartbreak", "terrif", "alarm", "panic", "outrag", "furious", "fury",
                  "chaos", "nightmare", "tragic", "horrif", "stagger", "plunge", "explosive", "dramatic", "stun",
                  "breakthrough", "revolution", "game-chang", "groundbreak", "unprecedented", "incredible",
                  "amazing", "remarkable", "milestone", "boom", "transform", "powerful", "exciting", "promising",
                  "soar", "slam", "blast", "fail", "scandal", "lawsuit", "sues", "sued", "banned", "crackdown",
                  "warns", "warning", "threat", "backlash", "critic", "controvers", "accus", "probe", "layoff",
                  "breach", "hack", "flaw", "antitrust"],
    "OPINIONATED": ["should", "must", "opinion", "editorial", "why ", "hype", "myth", "overrated", "mistake",
                    "wrong", "worst", "we need", "you need", "i think", "argue", "column"],
}


def keyword_baseline(text):
    t = " " + text.lower() + " "
    counts = {k: sum(1 for s in v if s in t) for k, v in LEX.items()}
    best = max(counts, key=counts.get)
    return best if counts[best] >= 1 else "NEUTRAL"


def make_pipeline(C=1.0):
    return Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, token_pattern=TOKEN_PATTERN, ngram_range=(1, 2),
                                  min_df=2, max_features=5000, sublinear_tf=True)),
        ("clf", LogisticRegression(C=C, max_iter=3000, class_weight="balanced")),
    ])


def load_data(path):
    texts, labels = [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            text, label = (row.get("text") or "").strip(), (row.get("label") or "").strip().upper()
            if text and label:
                texts.append(text)
                labels.append(label)
    return texts, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="labels.csv")
    ap.add_argument("--out", default="../tone_model.json")
    ap.add_argument("--report", default="report.txt")
    args = ap.parse_args()

    texts, labels = load_data(args.data)
    counts = Counter(labels)
    print(f"Loaded {len(texts)} labelled headlines: {dict(counts)}")
    if len(counts) < 2:
        sys.exit("Need at least two different labels.")
    if min(counts.values()) < 10:
        sys.exit("Each label needs at least 10 examples (aim for 40+). Keep labelling, then run this again.")

    X = np.array(texts, dtype=object)
    y = np.array(labels)
    grid = {"clf__C": [0.5, 1, 2, 5, 10, 20]}

    # Nested cross-validation: the inner loop picks C, the outer loop scores it on unseen data,
    # so the score is not inflated by tuning.
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model_cv = GridSearchCV(make_pipeline(), grid, scoring="f1_macro", cv=3)
    pred = cross_val_predict(model_cv, X, y, cv=outer)

    classes = sorted(set(labels))
    acc, f1 = accuracy_score(y, pred), f1_score(y, pred, average="macro")
    maj = DummyClassifier(strategy="most_frequent").fit(X, y).predict(X)
    kw = np.array([keyword_baseline(t) for t in texts])
    base = {
        "majority_class": (accuracy_score(y, maj), f1_score(y, maj, average="macro")),
        "keyword_rules": (accuracy_score(y, kw), f1_score(y, kw, average="macro")),
    }

    lines = []
    lines.append(f"Examples: {len(texts)}   Classes: {dict(counts)}")
    lines.append("")
    lines.append("Cross-validated results (5-fold, nested):")
    lines.append(f"  Your model         accuracy {acc:.3f}   macro-F1 {f1:.3f}")
    lines.append(f"  Keyword rules      accuracy {base['keyword_rules'][0]:.3f}   macro-F1 {base['keyword_rules'][1]:.3f}")
    lines.append(f"  Always-majority    accuracy {base['majority_class'][0]:.3f}   macro-F1 {base['majority_class'][1]:.3f}")
    lines.append("")
    lines.append(classification_report(y, pred, labels=classes, zero_division=0))
    lines.append("Confusion matrix (rows = true, columns = predicted), order: " + ", ".join(classes))
    lines.append(str(confusion_matrix(y, pred, labels=classes)))

    # Final model: fit on all the data
    search = GridSearchCV(make_pipeline(), grid, scoring="f1_macro", cv=3).fit(X, y)
    best = search.best_estimator_
    tfidf, clf = best.named_steps["tfidf"], best.named_steps["clf"]

    lines.append("")
    lines.append(f"Chosen regularisation C = {search.best_params_['clf__C']}")
    vocab = [None] * len(tfidf.vocabulary_)
    for term, i in tfidf.vocabulary_.items():
        vocab[i] = term
    coef, intercept = clf.coef_, clf.intercept_
    cls_names = [str(c) for c in clf.classes_]
    lines.append("")
    lines.append("Terms that push hardest toward each label:")
    rows = coef if len(cls_names) > 2 else np.vstack([-coef[0], coef[0]])
    for k, name in enumerate(cls_names):
        top = np.argsort(rows[k])[::-1][:10]
        lines.append(f"  {name}: " + ", ".join(vocab[i] for i in top))

    # The browser always uses softmax; convert the 2-class case to the same form.
    if len(cls_names) == 2:
        coef = np.vstack([-coef[0] / 2, coef[0] / 2])
        intercept = np.array([-intercept[0] / 2, intercept[0] / 2])

    model = {
        "version": 1,
        "classes": cls_names,
        "sublinear": True,
        "vocab": vocab,
        "idf": [round(float(v), 5) for v in tfidf.idf_],
        "coef": [[round(float(v), 4) for v in row] for row in coef],
        "intercept": [round(float(v), 4) for v in intercept],
        "meta": {
            "trained_on": date.today().isoformat(),
            "n_samples": len(texts),
            "class_counts": dict(counts),
            "cv_accuracy": round(float(acc), 4),
            "cv_macro_f1": round(float(f1), 4),
            "baseline_keyword_macro_f1": round(float(base["keyword_rules"][1]), 4),
            "baseline_majority_macro_f1": round(float(base["majority_class"][1]), 4),
            "best_C": search.best_params_["clf__C"],
        },
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(model, f, separators=(",", ":"))
    with open(args.report, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\nSaved model -> {args.out}  and report -> {args.report}")


if __name__ == "__main__":
    main()
