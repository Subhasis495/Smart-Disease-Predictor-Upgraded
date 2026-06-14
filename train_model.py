"""
train_model.py — Run this once to train and save the ML models.
Usage: python train_model.py --train data/training_data.csv --test data/test_data.csv
"""

import argparse
import pandas as pd
import numpy as np
import json
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

def train(train_path: str, test_path: str, out_dir: str = "models"):
    print("📂 Loading data...")
    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)

    # Drop empty columns
    train_df = train_df.drop(columns=[c for c in train_df.columns if "Unnamed" in c], errors="ignore")

    feature_cols = [c for c in train_df.columns if c != "prognosis"]
    X_train = train_df[feature_cols].values
    X_test  = test_df[feature_cols].values

    le = LabelEncoder()
    y_train = le.fit_transform(train_df["prognosis"])
    y_test  = le.transform(test_df["prognosis"])

    # ── Random Forest ──────────────────────────────────────────────────────────
    print("🌲 Training Random Forest (200 trees)...")
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(X_test))
    print(f"   ✅ RF Accuracy: {rf_acc*100:.2f}%")

    # ── SVM ───────────────────────────────────────────────────────────────────
    print("⚡ Training SVM (RBF kernel)...")
    svm = SVC(kernel="rbf", probability=True, random_state=42)
    svm.fit(X_train, y_train)
    svm_acc = accuracy_score(y_test, svm.predict(X_test))
    print(f"   ✅ SVM Accuracy: {svm_acc*100:.2f}%")

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(rf,  os.path.join(out_dir, "rf_model.pkl"))
    joblib.dump(svm, os.path.join(out_dir, "svm_model.pkl"))
    joblib.dump(le,  os.path.join(out_dir, "label_encoder.pkl"))

    meta = {
        "feature_cols":   feature_cols,
        "diseases":       le.classes_.tolist(),
        "rf_accuracy":    float(rf_acc),
        "svm_accuracy":   float(svm_acc),
        "n_train":        int(len(X_train)),
        "n_test":         int(len(X_test)),
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n🎉 All models saved to '{out_dir}/'")
    print(f"   Features : {len(feature_cols)}")
    print(f"   Diseases : {len(le.classes_)}")
    print(f"   Training : {len(X_train)} samples")

    # Full report
    print("\n📊 Classification Report (SVM):")
    print(classification_report(y_test, svm.predict(X_test),
                                 target_names=le.classes_, zero_division=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Smart Disease Predictor models")
    parser.add_argument("--train", default="data/training_data.csv")
    parser.add_argument("--test",  default="data/test_data.csv")
    parser.add_argument("--out",   default="models")
    args = parser.parse_args()
    train(args.train, args.test, args.out)
