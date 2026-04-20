# app_gui.py
import joblib
import tkinter as tk
from tkinter import ttk, messagebox

# ---------- ADD THIS BLOCK (same as training script) ----------
import re, string
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

# Ensure NLTK data is available
try:
    _ = stopwords.words("english")
except:
    nltk.download("punkt")
    nltk.download("stopwords")

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
    text = re.sub(r"http\S+|www\.\S+", " url ", text)
    text = re.sub(r"\d+", " number ", text)

    tokens = word_tokenize(text)
    out = []
    for t in tokens:
        if t.isalnum() and t not in STOP and t not in PUNCT:
            out.append(ps.stem(t))
    return out
# ---------- END PATCH BLOCK ----------

MODEL_PATH = "sms_spam_svm.joblib"

try:
    clf = joblib.load(MODEL_PATH)
except Exception as e:
    raise SystemExit(
        f"Could not load {MODEL_PATH}. Train first by running: python train_evaluate.py\n\n{e}"
    )

def predict_text():
    text = input_box.get("1.0", tk.END).strip()
    if not text:
        messagebox.showinfo("Info", "Please enter a message.")
        return
    proba = clf.predict_proba([text])[0, 1]  # probability of spam class
    pred = (proba >= 0.5)
    label = "SPAM" if pred else "HAM (not spam)"
    result_var.set(f"Prediction: {label}    |    Confidence: {proba:.2%}")
    result_lbl.configure(foreground=("#B00020" if pred else "#155724"))

# --- Simple Tkinter UI ---
root = tk.Tk()
root.title("SMS Spam Detector (SVM)")
root.geometry("620x380")
root.configure(bg="#f7f9fc")

title = ttk.Label(root, text="SMS Spam Detector", font=("Segoe UI", 18, "bold"))
title.pack(pady=10)

info = ttk.Label(root, text="Type/paste an SMS message below and click Predict.", font=("Segoe UI", 10))
info.pack()

input_box = tk.Text(root, height=8, width=68, wrap=tk.WORD, font=("Segoe UI", 10))
input_box.pack(padx=12, pady=8)

predict_btn = ttk.Button(root, text="Predict", command=predict_text)
predict_btn.pack(pady=5)

result_var = tk.StringVar(value="Prediction: —")
result_lbl = ttk.Label(root, textvariable=result_var, font=("Segoe UI", 11))
result_lbl.pack(pady=10)

root.mainloop()
