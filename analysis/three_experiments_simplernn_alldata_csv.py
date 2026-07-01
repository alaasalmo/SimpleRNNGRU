"""
THREE EXPERIMENTS — SimpleRNN sentiment classifier (Kaggle all-data.csv)
==========================================================================
Data source: Kaggle — all-data.csv (columns: label,text — no header row)
Labels are text strings: "negative", "neutral", "positive"
We map them to indices matching CLASS_NAMES below.

1) Dropout: compare dropout=0.45 vs NO dropout (0.0), find best epoch for each, plot both.
2) Class weights: best dropout from (1), compare WITH vs WITHOUT class weights, plot both.
3) Architecture: compare SimpleRNN vs Bidirectional SimpleRNN (best settings from above), plot both.

Each experiment saves its own PNG graph.
"""

import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter

# ----------------------------------------------------------------
# SETTINGS
# ----------------------------------------------------------------
SEED = 42
MAX_EPOCHS = 50
PATIENCE = 5
CLASS_NAMES = ["bearish", "bullish", "neutral"]

# maps the text labels found in all-data.csv to the CLASS_NAMES indices above
# (Kaggle financial-sentiment csv uses "negative"/"positive"/"neutral")
LABEL_TO_INDEX = {
    "negative": 0,   # bearish
    "positive": 1,   # bullish
    "neutral": 2,    # neutral
}

DATA_PATH = "all-data.csv"

HP = {
    "learning_rate": 0.0005,
    "vocab_size": 6000,
    "embedding_dim": 32,
    "rnn_units": 32,
    "max_len": 60,
    "dropout_rate": 0.45,   # default / baseline dropout
}

DROPOUT_SETTINGS_TO_TEST = [0.0, 0.45]   # experiment 1: no dropout vs dropout=0.45

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ==================================================================
# PART 1: LOAD DATA
# ==================================================================
def load_data():
    """
    Reads all-data.csv with columns: label,text (no header row).
    Splits into train/test (80/20, stratified) since the file has
    no separate test split provided.
    """
    df = pd.read_csv(DATA_PATH, header=None, names=["label", "text"],
                      encoding="latin-1")

    df["label"] = df["label"].str.strip().str.lower()
    df["text"] = df["text"].astype(str).str.strip()

    # drop rows with unknown labels or empty text
    df = df[df["label"].isin(LABEL_TO_INDEX.keys())]
    df = df[df["text"] != ""]

    df["label_idx"] = df["label"].map(LABEL_TO_INDEX)

    texts = df["text"].tolist()
    labels = df["label_idx"].astype(int).tolist()

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, stratify=labels, random_state=SEED
    )
    return train_texts, train_labels, test_texts, test_labels


# ==================================================================
# PART 2: VECTORIZATION
# ==================================================================
def build_vectorizer(texts):
    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=HP["vocab_size"],
        output_sequence_length=HP["max_len"],
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


# ==================================================================
# PART 3: MODELS
# ==================================================================
def build_model(dropout_rate=HP["dropout_rate"], bidirectional=False):
    """SimpleRNN classifier. Set bidirectional=True for BiSimpleRNN."""
    rnn_layer = tf.keras.layers.SimpleRNN(HP["rnn_units"])
    if bidirectional:
        rnn_layer = tf.keras.layers.Bidirectional(rnn_layer)

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(HP["vocab_size"], HP["embedding_dim"]),
        rnn_layer,
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(3, activation="softmax")
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=HP["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# ==================================================================
# PART 4: TRACK MACRO-F1 EVERY EPOCH + EARLY STOP AT BEST POINT
# ==================================================================
class TrackBestEpoch(tf.keras.callbacks.Callback):
    """
    Each epoch: predicts on validation set, computes macro-F1,
    remembers the best epoch's weights, and stops early if stuck.
    """
    def __init__(self, val_dataset, patience):
        super().__init__()
        self.val_dataset = val_dataset
        self.patience = patience
        self.f1_per_epoch = []
        self.best_f1 = -1
        self.best_epoch = 0
        self.best_weights = None
        self.epochs_without_improvement = 0

    def on_epoch_end(self, epoch, logs=None):
        true_labels, predicted_labels = [], []
        for batch_x, batch_y in self.val_dataset:
            predictions = self.model.predict(batch_x, verbose=0).argmax(axis=1)
            predicted_labels.extend(predictions.tolist())
            true_labels.extend(batch_y.numpy().tolist())

        current_f1 = f1_score(true_labels, predicted_labels, average="macro")
        self.f1_per_epoch.append(current_f1)

        if current_f1 > self.best_f1:
            self.best_f1 = current_f1
            self.best_epoch = epoch + 1
            self.best_weights = self.model.get_weights()
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        print(f"  Epoch {epoch+1}: val_macro_f1 = {current_f1:.4f}")

        if self.epochs_without_improvement >= self.patience:
            print(f"  No improvement for {self.patience} epochs. Stopping.")
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)


# ==================================================================
# PART 5: CLASS WEIGHTS
# ==================================================================
def get_class_weights(labels):
    classes = np.unique(labels)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    return {int(c): float(w) for c, w in zip(classes, weights)}


# ==================================================================
# PART 6: TRAIN + EVALUATE (one run)
# ==================================================================
def train_and_evaluate(train_texts, train_labels, test_texts, test_labels, vectorizer,
                        dropout_rate=HP["dropout_rate"], use_class_weights=False,
                        bidirectional=False, run_name=""):
    tr_texts, val_texts, tr_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=0.15,
        stratify=train_labels, random_state=SEED
    )

    train_ds = to_tf_dataset(tr_texts, tr_labels, vectorizer, shuffle=True)
    val_ds = to_tf_dataset(val_texts, val_labels, vectorizer, shuffle=False)
    test_ds = to_tf_dataset(test_texts, test_labels, vectorizer, shuffle=False)

    model = build_model(dropout_rate=dropout_rate, bidirectional=bidirectional)
    tracker = TrackBestEpoch(val_ds, patience=PATIENCE)

    class_weight_dict = get_class_weights(tr_labels) if use_class_weights else None
    if class_weight_dict:
        print(f"Class weights: {class_weight_dict}")

    print(f"\n--- {run_name} ---")
    print(f"dropout={dropout_rate}, class_weights={'ON' if use_class_weights else 'OFF'}, "
          f"bidirectional={bidirectional} (up to {MAX_EPOCHS} epochs, patience={PATIENCE})")

    model.fit(train_ds, epochs=MAX_EPOCHS, verbose=0,
              class_weight=class_weight_dict, callbacks=[tracker])

    true_labels, predicted_labels = [], []
    for batch_x, batch_y in test_ds:
        predictions = model.predict(batch_x, verbose=0).argmax(axis=1)
        predicted_labels.extend(predictions.tolist())
        true_labels.extend(batch_y.numpy().tolist())

    print(f"Best epoch: {tracker.best_epoch} (out of {len(tracker.f1_per_epoch)} trained)")
    print(classification_report(true_labels, predicted_labels,
                                 target_names=CLASS_NAMES, zero_division=0))

    return tracker


# ==================================================================
# PART 7: PLOTTING (shared helper)
# ==================================================================
def plot_comparison(results, title, filename):
    """results: list of (label, tracker) tuples."""
    plt.figure(figsize=(9, 5))
    colors = ["#1baf7a", "#2a78d6", "#e34948", "#a26fd4"]

    for (label, tracker), color in zip(results, colors):
        epochs = range(1, len(tracker.f1_per_epoch) + 1)
        plt.plot(epochs, tracker.f1_per_epoch, label=label, color=color)
        plt.scatter(tracker.best_epoch, tracker.f1_per_epoch[tracker.best_epoch - 1],
                    color=color, marker="*", s=150, zorder=5, edgecolors="black")

    plt.xlabel("Epoch")
    plt.ylabel("Validation Macro-F1")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()
    print(f"Saved: {filename}")


# ==================================================================
# RUN ALL THREE EXPERIMENTS
# ==================================================================
if __name__ == "__main__":
    train_texts, train_labels, test_texts, test_labels = load_data()
    print(f"Train: {len(train_texts)} | Test: {len(test_texts)}")
    print("Train class counts:", Counter(train_labels))

    vectorizer = build_vectorizer(train_texts)

    # ============================================================
    # EXPERIMENT 1: Dropout sweep (baseline SimpleRNN, no class weights)
    # ============================================================
    print(f"\n{'='*60}\nEXPERIMENT 1: DROPOUT vs NO DROPOUT\n{'='*60}")
    dropout_results = []
    for rate in DROPOUT_SETTINGS_TO_TEST:
        label = "no dropout" if rate == 0.0 else f"dropout={rate}"
        tracker = train_and_evaluate(
            train_texts, train_labels, test_texts, test_labels, vectorizer,
            dropout_rate=rate, use_class_weights=False, bidirectional=False,
            run_name=label
        )
        dropout_results.append((label, tracker))

    plot_comparison(
        dropout_results,
        title="Experiment 1: Macro-F1 — Dropout vs No Dropout\n(stars = each curve's best epoch)",
        filename="exp1_dropout_comparison.png"
    )

    print("\nEXPERIMENT 1 SUMMARY")
    for label, tracker in dropout_results:
        print(f"  {label:<15} best_macro_f1={tracker.best_f1:.4f}  best_epoch={tracker.best_epoch}")

    best_label, best_tracker = max(dropout_results, key=lambda x: x[1].best_f1)
    best_dropout = 0.0 if best_label == "no dropout" else float(best_label.split("=")[1])
    print(f"\nBest setting: {best_label} (macro-F1={best_tracker.best_f1:.4f})")

    # ============================================================
    # EXPERIMENT 2: Class weights ON vs OFF (using best dropout)
    # ============================================================
    print(f"\n{'='*60}\nEXPERIMENT 2: CLASS WEIGHTS (dropout fixed at {best_dropout})\n{'='*60}")

    tracker_no_weight = train_and_evaluate(
        train_texts, train_labels, test_texts, test_labels, vectorizer,
        dropout_rate=best_dropout, use_class_weights=False, bidirectional=False,
        run_name="Class weights OFF"
    )
    tracker_with_weight = train_and_evaluate(
        train_texts, train_labels, test_texts, test_labels, vectorizer,
        dropout_rate=best_dropout, use_class_weights=True, bidirectional=False,
        run_name="Class weights ON"
    )

    weight_results = [
        ("No class weights", tracker_no_weight),
        ("With class weights", tracker_with_weight),
    ]
    plot_comparison(
        weight_results,
        title=f"Experiment 2: Macro-F1 With vs Without Class Weights\n(dropout={best_dropout}, stars = best epoch)",
        filename="exp2_class_weight_comparison.png"
    )

    print("\nEXPERIMENT 2 SUMMARY")
    for label, tracker in weight_results:
        print(f"  {label:<20} best_macro_f1={tracker.best_f1:.4f}  best_epoch={tracker.best_epoch}")

    # pick best class-weight setting to carry into experiment 3
    use_weights_for_exp3 = tracker_with_weight.best_f1 > tracker_no_weight.best_f1
    print(f"\nUsing class_weights={'ON' if use_weights_for_exp3 else 'OFF'} for Experiment 3 "
          f"(whichever scored higher above)")

    # ============================================================
    # EXPERIMENT 3: SimpleRNN vs Bidirectional SimpleRNN
    # (best dropout + best class-weight setting from above)
    # ============================================================
    print(f"\n{'='*60}\nEXPERIMENT 3: SimpleRNN vs BiSimpleRNN\n{'='*60}")

    tracker_simple = train_and_evaluate(
        train_texts, train_labels, test_texts, test_labels, vectorizer,
        dropout_rate=best_dropout, use_class_weights=use_weights_for_exp3,
        bidirectional=False, run_name="SimpleRNN"
    )
    tracker_bi = train_and_evaluate(
        train_texts, train_labels, test_texts, test_labels, vectorizer,
        dropout_rate=best_dropout, use_class_weights=use_weights_for_exp3,
        bidirectional=True, run_name="Bidirectional SimpleRNN"
    )

    arch_results = [
        ("SimpleRNN", tracker_simple),
        ("BiSimpleRNN", tracker_bi),
    ]
    plot_comparison(
        arch_results,
        title=f"Experiment 3: SimpleRNN vs BiSimpleRNN\n(dropout={best_dropout}, "
              f"class_weights={'ON' if use_weights_for_exp3 else 'OFF'}, stars = best epoch)",
        filename="exp3_architecture_comparison.png"
    )

    print("\nEXPERIMENT 3 SUMMARY")
    for label, tracker in arch_results:
        print(f"  {label:<15} best_macro_f1={tracker.best_f1:.4f}  best_epoch={tracker.best_epoch}")

    # ============================================================
    # FINAL OVERALL SUMMARY
    # ============================================================
    print(f"\n{'='*60}\nFINAL OVERALL SUMMARY\n{'='*60}")
    print(f"Best dropout setting (Exp 1): {best_label}")
    print(f"Best class-weight setting:   {'ON' if use_weights_for_exp3 else 'OFF'}")
    print(f"SimpleRNN best macro-F1:     {tracker_simple.best_f1:.4f} (epoch {tracker_simple.best_epoch})")
    print(f"BiSimpleRNN best macro-F1:   {tracker_bi.best_f1:.4f} (epoch {tracker_bi.best_epoch})")
    print("\nSaved graphs:")
    print("  exp1_dropout_comparison.png")
    print("  exp2_class_weight_comparison.png")
    print("  exp3_architecture_comparison.png")
