import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from collections import Counter

SEED = 42
EPOCHS = 30      # max epochs; early stopping decides the real stopping point
PATIENCE = 5     # stop if combined score doesn't improve for this many epochs
CLASS_NAMES = ["negative", "neutral", "positive"]   # all-data.csv sentiment labels

# Fixed hyperparameters — used as-is for BOTH SimpleRNN and BiSimpleRNN below.
DEFAULTS = {
    "learning_rate": 0.001,
    "vocab_size": 6000,
    "embedding_dim": 64,
    "rnn_units": 64,
    "max_len": 60,
}

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ==================================================================
# DATA LOADING — Kaggle Financial PhraseBank (all-data.csv)
# ==================================================================
def load_data(csv_path="all-data.csv", test_size=0.15):
    """
    Loads the Financial PhraseBank dataset from all-data.csv.
    File has NO header and two columns: sentiment, text
    (sentiment is one of: negative, neutral, positive).
    Typically stored as latin-1, not utf-8.

    This CSV has no separate train/test split, so one is carved out
    here with a stratified, fixed-seed split.
    """
    candidates = [
        csv_path,
        f"/kaggle/input/sentiment-analysis-for-financial-news/{csv_path}",
        f"/kaggle/input/{csv_path}",
    ]
    resolved_path = next((p for p in candidates if os.path.exists(p)), csv_path)

    df = pd.read_csv(resolved_path, header=None, names=["sentiment", "text"],
                      encoding="latin-1")
    df["sentiment"] = df["sentiment"].astype(str).str.strip().str.lower()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""]

    label_map = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    df = df[df["sentiment"].isin(label_map)]
    texts = df["text"].tolist()
    labels = df["sentiment"].map(label_map).tolist()

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=test_size, stratify=labels, random_state=SEED
    )
    return train_texts, train_labels, test_texts, test_labels


def build_vectorizer(texts, vocab_size, max_len):
    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=vocab_size,
        output_sequence_length=max_len,
        standardize="lower_and_strip_punctuation"
    )
    vectorizer.adapt(texts)
    return vectorizer


def to_tf_dataset(texts, labels, vectorizer, shuffle=True, batch_size=32):
    ds = tf.data.Dataset.from_tensor_slices((texts, labels))
    if shuffle:
        ds = ds.shuffle(len(texts), seed=SEED)
    ds = ds.batch(batch_size)
    ds = ds.map(lambda x, y: (vectorizer(x), y))
    return ds.prefetch(tf.data.AUTOTUNE)


def print_label_distribution(labels, name):
    counts = Counter(labels)
    total = len(labels)
    parts = []
    for idx, class_name in enumerate(CLASS_NAMES):
        count = counts.get(idx, 0)
        pct = (count / total * 100) if total > 0 else 0.0
        parts.append(f"{class_name.capitalize()}: {count} ({pct:.1f}%)")
    print(f"{name} class counts -> " + " | ".join(parts))


# ==================================================================
# EVALUATE: accuracy + macro-F1 -> combined score
# ==================================================================
def evaluate_combined_score(model, dataset, true_labels):
    """
    Returns (accuracy, macro_f1, combined) where combined = (accuracy + macro_f1) / 2.
    This is the metric early stopping monitors, matching how models are
    ultimately judged (not val_loss, and not macro-F1 alone).
    """
    predicted_labels = model.predict(dataset, verbose=0).argmax(axis=1)
    acc = accuracy_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels, average="macro")
    return acc, f1, (acc + f1) / 2


def train_with_early_stopping(model, train_ds, val_ds, val_labels, max_epochs, patience):
    """
    Trains one epoch at a time, monitoring the combined (accuracy + macro-F1)/2
    score on the validation set. Keeps the best-performing weights and stops
    early if there's no improvement for `patience` epochs in a row.
    Returns (best_combined, best_epoch, epochs_trained, val_history), where
    val_history is a list of the combined score at each epoch trained.
    """
    best_combined = -1
    best_epoch = 0
    best_weights = None
    epochs_without_improvement = 0
    epochs_trained = 0
    val_history = []

    for epoch in range(max_epochs):
        model.fit(train_ds, epochs=1, verbose=0)
        epochs_trained = epoch + 1

        val_acc, val_f1, val_combined = evaluate_combined_score(model, val_ds, val_labels)
        val_history.append(val_combined)

        if val_combined > best_combined:
            best_combined = val_combined
            best_epoch = epochs_trained
            best_weights = model.get_weights()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(f"  epoch {epochs_trained:>2} | val_acc={val_acc:.4f} val_f1={val_f1:.4f} "
              f"val_combined={val_combined:.4f} "
              f"{'<-- best' if epochs_without_improvement == 0 else ''}")

        if epochs_without_improvement >= patience:
            break

    if best_weights is not None:
        model.set_weights(best_weights)

    return best_combined, best_epoch, epochs_trained, val_history


# ==================================================================
# MODEL — SimpleRNN or BiSimpleRNN, selected via `bidirectional` flag
# ==================================================================
def build_model(vocab_size, embedding_dim, rnn_units, learning_rate, bidirectional):
    """
    bidirectional=False -> plain SimpleRNN (reads left-to-right only)
    bidirectional=True  -> Bidirectional(SimpleRNN) (reads both directions,
                            concatenates forward+backward outputs, doubling
                            the effective output size to rnn_units*2)
    """
    rnn_layer = tf.keras.layers.SimpleRNN(rnn_units)
    if bidirectional:
        rnn_layer = tf.keras.layers.Bidirectional(rnn_layer)

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim),
        rnn_layer,
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(3, activation="softmax")
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# ==================================================================
# TRAIN + EVALUATE ONE MODEL VARIANT
# ==================================================================
def run_one_variant(tr_texts, tr_labels, val_texts, val_labels, test_texts, test_labels,
                     vectorizer, config, bidirectional, label):
    train_ds = to_tf_dataset(tr_texts, tr_labels, vectorizer, shuffle=True)
    val_ds = to_tf_dataset(val_texts, val_labels, vectorizer, shuffle=False)
    test_ds = to_tf_dataset(test_texts, test_labels, vectorizer, shuffle=False)

    print(f"\n=== {label} ===")
    model = build_model(config["vocab_size"], config["embedding_dim"],
                         config["rnn_units"], config["learning_rate"],
                         bidirectional=bidirectional)

    best_combined, best_epoch, epochs_trained, val_history = train_with_early_stopping(
        model, train_ds, val_ds, val_labels, max_epochs=EPOCHS, patience=PATIENCE
    )

    test_acc, test_f1, test_combined = evaluate_combined_score(model, test_ds, test_labels)
    predicted_labels = model.predict(test_ds, verbose=0).argmax(axis=1)
    per_class_f1 = f1_score(test_labels, predicted_labels, average=None)

    per_class_str = " | ".join(
        f"{name.capitalize()} F1: {f1:.3f}" for name, f1 in zip(CLASS_NAMES, per_class_f1)
    )
    print(f"{label:<15} | Acc: {test_acc*100:5.2f}% | Macro-F1: {test_f1:.4f} | "
          f"Combined: {test_combined:.4f} | {per_class_str} | "
          f"Stopped epoch: {epochs_trained} (best: {best_epoch})")

    return {"label": label, "bidirectional": bidirectional,
            "test_acc": test_acc, "test_macro_f1": test_f1,
            "test_combined": test_combined, "per_class_f1": per_class_f1,
            "epochs_trained": epochs_trained, "best_epoch": best_epoch,
            "val_history": val_history,
            "test_true_labels": test_labels, "test_pred_labels": predicted_labels}


# ==================================================================
# PLOT: validation combined-score curves for both models,
# with each model's best epoch marked
# ==================================================================
def plot_results(results, filename="simplernn_vs_bisimplernn_kaggle.png"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, filename)

    plt.figure(figsize=(9, 5))
    colors = {"SimpleRNN": "#2a78d6", "BiSimpleRNN": "#d6572a"}

    for r in results:
        epochs = list(range(1, len(r["val_history"]) + 1))
        color = colors.get(r["label"], None)
        plt.plot(epochs, r["val_history"], marker="o", label=r["label"], color=color)

        best_epoch = r["best_epoch"]
        best_score = r["val_history"][best_epoch - 1]
        plt.scatter([best_epoch], [best_score], s=160, marker="*",
                    color=color, edgecolors="black", zorder=5)
        plt.annotate(f"best: epoch {best_epoch}\n({best_score:.3f})",
                     (best_epoch, best_score),
                     textcoords="offset points", xytext=(0, 12),
                     ha="center", fontsize=9)

    plt.xlabel("Epoch")
    plt.ylabel("Validation Combined Score  (Accuracy + Macro-F1) / 2")
    plt.title("SimpleRNN vs BiSimpleRNN — Validation Combined Score per Epoch\nKaggle Financial Phrasebank Dataset")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved: {save_path}")


# ==================================================================
# PRINT: confusion matrix for each model, as plain text tables
# ==================================================================
def print_confusion_matrix(result):
    """
    Prints a row-normalized confusion matrix as a plain text table:
    rows = true label, columns = predicted label. Each cell shows
    count and row percentage, e.g. "106 (30.5%)".
    """
    cm = confusion_matrix(result["test_true_labels"], result["test_pred_labels"],
                           labels=list(range(len(CLASS_NAMES))))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    col_names = [c.capitalize() for c in CLASS_NAMES]
    row_label_w = max(len(c) for c in col_names) + 6  # room for "True: "
    cell_w = 16

    header = f"{result['label']}  (Acc: {result['test_acc']*100:.2f}%  " \
             f"Macro-F1: {result['test_macro_f1']:.4f})"
    print("\n" + header)
    print("-" * len(header))

    # Column header row
    print(" " * row_label_w + "".join(f"{c:^{cell_w}}" for c in col_names) +
          f"{'Total':^{cell_w}}")

    for i, true_name in enumerate(col_names):
        row_label = f"True: {true_name}"
        cells = "".join(
            f"{f'{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)':^{cell_w}}"
            for j in range(len(col_names))
        )
        total = cm[i].sum()
        print(f"{row_label:<{row_label_w}}" + cells + f"{total:^{cell_w}}")


def print_confusion_matrices(results):
    print("\n" + "=" * 70)
    print("Confusion Matrices (row-normalized) — Test Set")
    print("=" * 70)
    for r in results:
        print_confusion_matrix(r)


# ==================================================================
# RUN: SimpleRNN vs BiSimpleRNN, same fixed hyperparameters
# ==================================================================
if __name__ == "__main__":
    train_texts, train_labels, test_texts, test_labels = load_data()
    print(f"Train: {len(train_texts)} | Test: {len(test_texts)}")
    print_label_distribution(train_labels, "Train")
    print_label_distribution(test_labels, "Test")

    tr_texts, val_texts, tr_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=0.15,
        stratify=train_labels, random_state=SEED
    )
    print_label_distribution(tr_labels, "Train split")
    print_label_distribution(val_labels, "Val split")

    vectorizer = build_vectorizer(tr_texts, DEFAULTS["vocab_size"], DEFAULTS["max_len"])

    simplernn_result = run_one_variant(
        tr_texts, tr_labels, val_texts, val_labels, test_texts, test_labels,
        vectorizer, DEFAULTS, bidirectional=False, label="SimpleRNN"
    )

    bisimplernn_result = run_one_variant(
        tr_texts, tr_labels, val_texts, val_labels, test_texts, test_labels,
        vectorizer, DEFAULTS, bidirectional=True, label="BiSimpleRNN"
    )

    plot_results([simplernn_result, bisimplernn_result])
    print_confusion_matrices([simplernn_result, bisimplernn_result])

    # ---------------------------------------------------------
    # COMPARISON SUMMARY
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("SimpleRNN vs BiSimpleRNN — Comparison (fixed hyperparameters)")
    print("=" * 70)
    print(f"Hyperparameters used: {DEFAULTS}")
    print()

    header = f"{'Model':<14}{'Acc':>8}{'Macro-F1':>10}{'Combined':>10}{'Best Epoch':>12}{'Stopped':>10}"
    print(header)
    print("-" * len(header))
    for r in (simplernn_result, bisimplernn_result):
        print(f"{r['label']:<14}{r['test_acc']*100:>7.2f}%{r['test_macro_f1']:>10.4f}"
              f"{r['test_combined']:>10.4f}{r['best_epoch']:>12}{r['epochs_trained']:>10}")

    winner = max((simplernn_result, bisimplernn_result), key=lambda r: r["test_combined"])
    print(f"\nBetter model (by combined score): {winner['label']} "
          f"(combined={winner['test_combined']:.4f}, best epoch={winner['best_epoch']})")

    print("\n" + "-" * 70)
    print("Dataset label distribution")
    print("-" * 70)
    print_label_distribution(train_labels, "Train")
    print_label_distribution(test_labels, "Test")
