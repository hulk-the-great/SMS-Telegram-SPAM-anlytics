# train_evaluate.py
import os
import re
import string
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report
)
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer

# --- NLTK setup ---
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

# Download once if missing
try:
    _ = stopwords.words("english")
except:
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')

ps = PorterStemmer()
STOP = set(stopwords.words("english"))
PUNCT = set(string.punctuation)

def clean_and_tokenize(text: str):
    """
    Lowercase → tokenize → keep alphanumeric tokens →
    remove stopwords/punctuation → Porter stem → return tokens.
    """
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)

    text = text.lower()
    # normalize urls & numbers a bit (optional but useful)
    text = re.sub(r"http\S+|www\.\S+", " url ", text)
    text = re.sub(r"\d+", " number ", text)

    tokens = word_tokenize(text)
    out = []
    for t in tokens:
        if t.isalnum() and t not in STOP and t not in PUNCT:
            out.append(ps.stem(t))
    return out

# === Load data ===
CSV_PATH = "spam.csv"   # Expect columns: Label, EmailText
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(
        f"Could not find {CSV_PATH}. Place your dataset in this folder "
        "with columns: Label (ham/spam) and EmailText."
    )

df = pd.read_csv(CSV_PATH)

# Keep just the required columns
expected_cols = {"Label", "EmailText"}
missing = expected_cols - set(df.columns)
if missing:
    raise ValueError(f"CSV is missing columns: {missing}. Found: {list(df.columns)}")

# Drop duplicates and NAs
df = df.drop_duplicates(subset=["EmailText"]).dropna(subset=["EmailText", "Label"]).copy()

# Encode labels: ham→0, spam→1
le = LabelEncoder()
df["y"] = le.fit_transform(df["Label"])  # {'ham':0, 'spam':1}

X = df["EmailText"].values
y = df["y"].values

# === Build pipeline ===
# We put TF-IDF + SVM into a single Pipeline so the exact preprocessing
# used at train time is saved and reused in the GUI.
pipeline = Pipeline(
    steps=[
        ("tfidf", TfidfVectorizer(
            tokenizer=clean_and_tokenize,      # custom NLTK tokenizer
            preprocessor=None,
            lowercase=False,                    # we lowercase inside the tokenizer
            ngram_range=(1, 2),                 # unigrams + bigrams (works well on SMS)
            min_df=2
        )),
        ("svm", SVC(kernel="linear", probability=True, class_weight="balanced"))
    ]
)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Optional: light hyperparameter search (kept small for demo speed)
param_grid = {
    "svm__C": [0.5, 1.0, 2.0]
}
grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=3,
    n_jobs=-1,
    verbose=1
)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
print(f"\nBest params: {grid.best_params_}")

# === Evaluate ===
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]  # for ROC/PR

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n=== Test Metrics ===")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["ham","spam"]))

# Confusion matrix plot
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(4, 3.6))
ax.imshow(cm, cmap="Blues")
ax.set_title("Confusion Matrix (Test)")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_xticks([0,1]); ax.set_yticks([0,1])
ax.set_xticklabels(["ham","spam"]); ax.set_yticklabels(["ham","spam"])
for (i, j), v in np.ndenumerate(cm):
    ax.text(j, i, str(v), ha="center", va="center", fontsize=11)
plt.tight_layout()
plt.savefig("figure_confusion_matrix.png", dpi=200)
plt.close()

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(4.5,3.6))
plt.plot(fpr, tpr, label=f"SVM (AUC={roc_auc:.2f})")
plt.plot([0,1], [0,1], "--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Test)")
plt.legend()
plt.tight_layout()
plt.savefig("figure_roc.png", dpi=200)
plt.close()

# Save the pipeline + label encoder
joblib.dump(best_model, "sms_spam_svm.joblib")
joblib.dump(le, "label_encoder.joblib")

print("\nSaved model to sms_spam_svm.joblib")
print("Saved label encoder to label_encoder.joblib")
print("Saved figures: figure_confusion_matrix.png, figure_roc.png")
