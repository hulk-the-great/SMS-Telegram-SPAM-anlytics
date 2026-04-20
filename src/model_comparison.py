from pathlib import Path
import time
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from text_utils import clean_and_tokenize

BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = BASE_DIR / "data" / "processed" / "featured_messages.csv"
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR = BASE_DIR / "models"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_PATH)

# Keep only needed columns
df = df[["message_id", "source_dataset", "label", "raw_text"]].copy()
df["label_num"] = df["label"].map({"ham": 0, "spam": 1})

X = df["raw_text"]
y = df["label_num"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

models = {
    "naive_bayes": MultinomialNB(),
    "logistic_regression": LogisticRegression(max_iter=1000),
    "linear_svm": SVC(kernel="linear", probability=True, random_state=42)
}

results = []
best_model_name = None
best_model = None
best_f1 = -1

for model_name, clf in models.items():
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            tokenizer=clean_and_tokenize,
            lowercase=False,
            ngram_range=(1, 2),
            max_features=10000
        )),
        ("clf", clf)
    ])

    start_train = time.time()
    pipeline.fit(X_train, y_train)
    train_time = time.time() - start_train

    start_pred = time.time()
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    inference_time = time.time() - start_pred

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)

    results.append({
        "model_name": model_name,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "train_time_sec": round(train_time, 4),
        "inference_time_sec": round(inference_time, 4)
    })

    print(f"\nModel: {model_name}")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")
    print(f"Train time (s): {train_time:.4f}")
    print(f"Inference time (s): {inference_time:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        best_model_name = model_name
        best_model = pipeline

results_df = pd.DataFrame(results).sort_values(by="f1_score", ascending=False)
results_path = REPORTS_DIR / "model_comparison_results.csv"
results_df.to_csv(results_path, index=False)

best_model_path = MODELS_DIR / f"best_model_{best_model_name}.joblib"
joblib.dump(best_model, best_model_path)

print("\nBest model:", best_model_name)
print("Saved best model to:", best_model_path)
print("Saved comparison results to:", results_path)
print("\nFinal comparison table:")
print(results_df.to_string(index=False))