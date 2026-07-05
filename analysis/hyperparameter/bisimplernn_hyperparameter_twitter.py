import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # silence TF C++ INFO/WARNING/ERROR logs

import warnings
warnings.filterwarnings("ignore")  # silence Python-level UserWarnings (e.g. shuffle=True ignored)

import random
import numpy as np
import tensorflow as tf
from datasets import load_dataset
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from collections import Counter

SEED = 42
EPOCHS = 30      # SimpleRNN converges faster but we give it enough room
PATIENCE = 5     # stop if macro-F1 doesn't improve for this many epochs
CLASS_NAMES = ["bearish", "bullish", "neutral"]

# Starting defaults - only ONE of these changes per test below
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
# DATA LOADING
# ==================================================================
def load_data():
    dataset = load_dataset("zeroshot/twitter-financial-news-sentiment")

    def extract(split):
        texts = [item["text"].strip() for item in dataset[split] if item["text"].strip()]
        labels = [int(item["label"]) for item in dataset[split] if item["text"].strip()]
        return texts, labels

    train_texts, train_labels = extract("train")
    test_texts, test_labels = extract("validation")
    return train_texts, train_labels, test_texts, test_labels


def build_vectorizer(texts, vocab_size, max_len):
    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=vocab_size,
        output_sequence_length=max_len,
        standardize="lower_and_strip_punctuation"
    )
    vectorizer.adapt(texts)
    return vectorizer


_VECTORIZER_CACHE = {}


def get_vectorizer(texts, vocab_size, max_len):
    """
    Reuses an already-adapted vectorizer for a given (vocab_size, max_len)
    instead of rebuilding + re-adapting it every run. Most sweeps only
    change one hyperparameter (learning rate, embedding dim, RNN units)
    while vocab_size/max_len stay fixed, so this skips a lot of repeat
    vocabulary-building work.
    """
    key = (vocab_size, max_len)
    if key not in _VECTORIZER_CACHE:
        _VECTORIZER_CACHE[key] = build_vectorizer(texts, vocab_size, max_len)
    return _VECTORIZER_CACHE[key]


def to_tf_dataset(texts, labels, vectorizer, shuffle=True, batch_size=32):
    ds = tf.data.Dataset.from_tensor_slices((texts, labels))
    if shuffle:
        ds = ds.shuffle(len(texts), seed=SEED)
    ds = ds.batch(batch_size)
    ds = ds.map(lambda x, y: (vectorizer(x), y))
    return ds.prefetch(tf.data.AUTOTUNE)


def print_label_distribution(labels, name):
    """
    Prints label counts using CLASS_NAMES (e.g. 'Bearish') along with
    each class's percentage share, instead of raw numeric indices.
    """
    counts = Counter(labels)
    total = len(labels)
    parts = []
    for idx, class_name in enumerate(CLASS_NAMES):
        count = counts.get(idx, 0)
        pct = (count / total * 100) if total > 0 else 0.0
        parts.append(f"{class_name.capitalize()}: {count} ({pct:.1f}%)")
    print(f"{name} class counts -> " + " | ".join(parts))


def evaluate_macro_f1(model, dataset, true_labels):
    """
    Macro-F1 for `model` on `dataset`. `true_labels` is the plain label
    list used to build `dataset` (safe to reuse directly since the
    dataset is never shuffled) — this avoids re-walking the dataset
    batch-by-batch just to recover labels we already have.
    """
    predicted_labels = model.predict(dataset, verbose=0).argmax(axis=1)
    return f1_score(true_labels, predicted_labels, average="macro")


def train_with_early_stopping(model, train_ds, val_ds, val_labels, max_epochs, patience):
    """
    Trains one epoch at a time and monitors validation macro-F1 directly
    (not val_loss), so the stopping point matches the metric we actually
    judge configurations by. Keeps the best-performing weights and stops
    early if there's no improvement for `patience` epochs in a row.
    Returns (best_f1, best_epoch, epochs_trained).
    """
    best_f1 = -1
    best_epoch = 0
    best_weights = None
    epochs_without_improvement = 0
    epochs_trained = 0

    for epoch in range(max_epochs):
        model.fit(train_ds, epochs=1, verbose=0, shuffle=False)
        epochs_trained = epoch + 1

        current_f1 = evaluate_macro_f1(model, val_ds, val_labels)

        if current_f1 > best_f1:
            best_f1 = current_f1
            best_epoch = epochs_trained
            best_weights = model.get_weights()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # print every epoch so you can see training is actually progressing
        print(f"  epoch {epochs_trained:>2} | val_macro_f1={current_f1:.4f} "
              f"{'<-- best' if epochs_without_improvement == 0 else ''}")

        if epochs_without_improvement >= patience:
            break

    if best_weights is not None:
        model.set_weights(best_weights)

    return best_f1, best_epoch, epochs_trained


def build_model(vocab_size, embedding_dim, rnn_units, learning_rate):
    """
    BiSimpleRNN model: text -> embedding -> Bidirectional SimpleRNN -> 3 classes.

    Bidirectional wraps the SimpleRNN so it reads the sequence twice:
      - Forward pass  (left to right, word 1 -> word N)
      - Backward pass (right to left, word N -> word 1)
    Both outputs are concatenated, giving the Dense layers a richer
    representation of each tweet than a one-direction SimpleRNN would
    produce. This doubles the effective output size of the RNN layer
    (rnn_units*2).
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim),
        tf.keras.layers.Bidirectional(tf.keras.layers.SimpleRNN(rnn_units)),
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
# TRAIN + EVALUATE ONE CONFIGURATION
# ==================================================================
def run_one_config(tr_texts, tr_labels, val_texts, val_labels, test_texts, test_labels,
                    config, label):
    """Train and evaluate one hyperparameter configuration. Returns a
    plain dict with the results - no plotting, no per-epoch history kept."""
    vectorizer = get_vectorizer(tr_texts, config["vocab_size"], config["max_len"])

    train_ds = to_tf_dataset(tr_texts, tr_labels, vectorizer, shuffle=True)
    val_ds = to_tf_dataset(val_texts, val_labels, vectorizer, shuffle=False)
    test_ds = to_tf_dataset(test_texts, test_labels, vectorizer, shuffle=False)

    model = build_model(config["vocab_size"], config["embedding_dim"],
                         config["rnn_units"], config["learning_rate"])

    best_f1, best_epoch, epochs_trained = train_with_early_stopping(
        model, train_ds, val_ds, val_labels, max_epochs=EPOCHS, patience=PATIENCE
    )

    predicted_labels = model.predict(test_ds, verbose=0).argmax(axis=1)
    test_acc = accuracy_score(test_labels, predicted_labels)
    test_macro_f1 = f1_score(test_labels, predicted_labels, average="macro")
    per_class_f1 = f1_score(test_labels, predicted_labels, average=None)

    print(f"{label:<15} | Acc: {test_acc*100:5.2f}% | Macro-F1: {test_macro_f1:.4f} | "
          f"Bearish F1: {per_class_f1[0]:.3f} | Bullish F1: {per_class_f1[1]:.3f} | "
          f"Neutral F1: {per_class_f1[2]:.3f} | "
          f"Stopped epoch: {epochs_trained} (best: {best_epoch})")

    return {"label": label, "config": dict(config),
            "test_acc": test_acc, "test_macro_f1": test_macro_f1,
            "per_class_f1": per_class_f1,
            "epochs_trained": epochs_trained,
            "best_epoch": best_epoch}


def print_summary(results, param_name):
    """Plain text summary table - picks and returns the best config."""
    print(f"\n--- {param_name} sweep results ---")
    best = max(results, key=lambda x: x["test_macro_f1"])
    print(f"Best: {best['label']}  "
          f"(Test Macro-F1={best['test_macro_f1']:.4f}, Test Acc={best['test_acc']*100:.2f}%, "
          f"stopped epoch {best['epochs_trained']}, best epoch {best['best_epoch']})\n")
    return best


# ==================================================================
# RUN: test each hyperparameter, one at a time, locking in the winner
# ==================================================================
if __name__ == "__main__":
    train_texts, train_labels, test_texts, test_labels = load_data()
    print(f"Train: {len(train_texts)} | Test: {len(test_texts)}")
    print_label_distribution(train_labels, "Train")
    print_label_distribution(test_labels, "Test")

    # Split ONCE and reuse for every run below — every run used the exact
    # same (deterministic, fixed-seed) split anyway, so splitting inside
    # each run just repeated identical work 14 times for no benefit.
    tr_texts, val_texts, tr_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=0.15,
        stratify=train_labels, random_state=SEED
    )
    print_label_distribution(tr_labels, "Train split")
    print_label_distribution(val_labels, "Val split")

    # ---------------------------------------------------------
    # TEST 1: Learning rate
    # ---------------------------------------------------------
    print("\n=== Learning Rate ===")
    lr_results = []
    for lr in [0.0005, 0.001, 0.002, 0.005]:   # added 0.005 — previous range was too low
        config = dict(DEFAULTS)
        config["learning_rate"] = lr
        lr_results.append(run_one_config(tr_texts, tr_labels, val_texts, val_labels,
                                          test_texts, test_labels, config, f"lr={lr}"))
    best_lr = print_summary(lr_results, "Learning Rate")
    DEFAULTS["learning_rate"] = best_lr["config"]["learning_rate"]

    # ---------------------------------------------------------
    # TEST 2: Vocab size
    # ---------------------------------------------------------
    print("=== Vocab Size ===")
    vocab_results = []
    for vs in [6000, 10000, 15000]:
        config = dict(DEFAULTS)
        config["vocab_size"] = vs
        vocab_results.append(run_one_config(tr_texts, tr_labels, val_texts, val_labels,
                                             test_texts, test_labels, config, f"vocab={vs}"))
    best_vocab = print_summary(vocab_results, "Vocab Size")
    DEFAULTS["vocab_size"] = best_vocab["config"]["vocab_size"]

    # ---------------------------------------------------------
    # TEST 3: Embedding dimension
    # ---------------------------------------------------------
    print("=== Embedding Dimension ===")
    embed_results = []
    for ed in [32, 64, 128]:
        config = dict(DEFAULTS)
        config["embedding_dim"] = ed
        embed_results.append(run_one_config(tr_texts, tr_labels, val_texts, val_labels,
                                             test_texts, test_labels, config, f"embed_dim={ed}"))
    best_embed = print_summary(embed_results, "Embedding Dimension")
    DEFAULTS["embedding_dim"] = best_embed["config"]["embedding_dim"]

    # ---------------------------------------------------------
    # TEST 4: RNN units
    # ---------------------------------------------------------
    print("=== RNN Units ===")
    units_results = []
    for u in [32, 64, 128]:
        config = dict(DEFAULTS)
        config["rnn_units"] = u
        units_results.append(run_one_config(tr_texts, tr_labels, val_texts, val_labels,
                                             test_texts, test_labels, config, f"units={u}"))
    best_units = print_summary(units_results, "RNN Units")
    DEFAULTS["rnn_units"] = best_units["config"]["rnn_units"]

    # ---------------------------------------------------------
    # FINAL: best combination found across all sweeps
    # ---------------------------------------------------------
    print("=== FINAL BEST BISIMPLERNN HYPERPARAMETER COMBINATION ===")
    print(DEFAULTS)
    final_result = run_one_config(tr_texts, tr_labels, val_texts, val_labels,
                                   test_texts, test_labels, DEFAULTS, "FINAL")
    print(f"\nFinal Test Accuracy : {final_result['test_acc']*100:.2f}%")
    print(f"Final Test Macro-F1 : {final_result['test_macro_f1']:.4f}")
    print(f"Stopped epoch       : {final_result['epochs_trained']} "
          f"(best epoch: {final_result['best_epoch']})")
    print(f"Best epoch          : {final_result['best_epoch']}")

    # ---------------------------------------------------------
    # MENTION: one-glance summary of every best value found
    # ---------------------------------------------------------
    sweeps_info = [
        ("Learning Rate",       "learning_rate", [0.0005, 0.001, 0.002, 0.005]),
        ("Vocab Size",          "vocab_size",    [6000, 10000, 15000]),
        ("Embedding Dimension", "embedding_dim", [32, 64, 128]),
        ("RNN Units",           "rnn_units",     [32, 64, 128]),
    ]
    print("\n" + "=" * 60)
    print("MENTION — Best BiSimpleRNN Hyperparameters Found")
    print("=" * 60)
    for name, key, values in sweeps_info:
        best_val = DEFAULTS[key]
        print(f"Best {name} {values}: {best_val}")
    print(f"\nFinal combined BiSimpleRNN -> Accuracy: {final_result['test_acc']*100:.2f}%  "
          f"Macro-F1: {final_result['test_macro_f1']:.4f}")
    print(f"Best epoch (FINAL config): {final_result['best_epoch']}")

    print("\n" + "-" * 60)
    print("Dataset label distribution")
    print("-" * 60)
    print_label_distribution(train_labels, "Train")
    print_label_distribution(test_labels, "Test")