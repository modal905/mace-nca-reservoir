"""
MNIST Reservoir Classification using NCA features.

Usage:
    # After running reservoir_mnist_make_dataset.py with a checkpoint:
    python reservoir_mnist_classify.py --train_csv mnist_x_train_v2.csv --test_csv mnist_x_test_v2.csv --label baseline

    # Compare two checkpoints directly:
    python reservoir_mnist_classify.py \
        --train_csv results/mnist_x_train_baseline.csv \
        --test_csv  results/mnist_x_test_baseline.csv  \
        --label baseline

    # Raw-pixel baseline (no CSV needed):
    python reservoir_mnist_classify.py --raw_baseline
"""
import numpy as np
import tensorflow as tf
import argparse
import os
import time
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report


def load_mnist_labels():
    """Load binarized MNIST train/test labels only."""
    (_, y_train), (_, y_test) = tf.keras.datasets.mnist.load_data()
    return np.squeeze(y_train), np.squeeze(y_test)


def load_nca_features(train_csv, test_csv):
    """Load NCA reservoir feature CSVs produced by reservoir_mnist_make_dataset.py."""
    import pandas as pd
    print(f"Loading train features: {train_csv}")
    x_train = pd.read_csv(train_csv, header=None).values.astype(np.float32)
    print(f"  shape: {x_train.shape}")

    print(f"Loading test features:  {test_csv}")
    x_test = pd.read_csv(test_csv, header=None).values.astype(np.float32)
    print(f"  shape: {x_test.shape}")

    return x_train, x_test


def load_raw_pixels():
    """Raw binarized MNIST pixels — comparison baseline, no NCA involved."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = ((x_train / 255.0) > 0.5).astype(np.float32).reshape(len(x_train), -1)
    x_test  = ((x_test  / 255.0) > 0.5).astype(np.float32).reshape(len(x_test),  -1)
    return x_train, np.squeeze(y_train), x_test, np.squeeze(y_test)


def train_and_evaluate(x_train, y_train, x_test, y_test, label=""):
    print(f"\n── Training LinearSVC [{label}] ─────────────────────────────")
    print(f"   Train: {x_train.shape}  |  Test: {x_test.shape}")
    t0 = time.time()
    clf = LinearSVC(max_iter=5000)
    clf.fit(x_train, y_train)
    train_time = time.time() - t0
    print(f"   Fit time: {train_time:.1f}s")

    y_pred = clf.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"   Test accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(classification_report(y_test, y_pred, digits=4))
    return acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", default="mnist_x_train_v2.csv",
                        help="Path to NCA reservoir train features CSV")
    parser.add_argument("--test_csv",  default="mnist_x_test_v2.csv",
                        help="Path to NCA reservoir test features CSV")
    parser.add_argument("--label",     default="NCA reservoir",
                        help="Label for this run (e.g. 'baseline' or 'conserving')")
    parser.add_argument("--raw_baseline", action="store_true",
                        help="Also run raw-pixel LinearSVC baseline for comparison")
    args = parser.parse_args()

    results = {}

    # ── NCA reservoir features ─────────────────────────────────────────────
    if os.path.exists(args.train_csv) and os.path.exists(args.test_csv):
        x_train, x_test = load_nca_features(args.train_csv, args.test_csv)
        y_train, y_test = load_mnist_labels()

        # Sanity check: CSV rows must match MNIST sample count
        assert len(x_train) == len(y_train), \
            f"Train size mismatch: CSV has {len(x_train)} rows, MNIST has {len(y_train)}"
        assert len(x_test) == len(y_test), \
            f"Test size mismatch: CSV has {len(x_test)} rows, MNIST has {len(y_test)}"

        acc_nca = train_and_evaluate(x_train, y_train, x_test, y_test, label=args.label)
        results[args.label] = acc_nca
    else:
        print(f"CSV not found: {args.train_csv} / {args.test_csv}")
        print("Run reservoir_mnist_make_dataset.py first.")

    # ── Raw-pixel baseline ─────────────────────────────────────────────────
    if args.raw_baseline:
        x_train_raw, y_train_raw, x_test_raw, y_test_raw = load_raw_pixels()
        acc_raw = train_and_evaluate(x_train_raw, y_train_raw,
                                     x_test_raw,  y_test_raw,
                                     label="raw pixels (baseline)")
        results["raw_pixels"] = acc_raw

    # ── Summary ────────────────────────────────────────────────────────────
    if len(results) > 1:
        print("\n── Summary ───────────────────────────────────────────────────")
        for label, acc in results.items():
            print(f"   {label:<30} {acc:.4f}  ({acc*100:.2f}%)")
