import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # silence TF C++ INFO/WARNING/ERROR logs

import warnings
warnings.filterwarnings("ignore")  # silence Python-level UserWarnings (e.g. shuffle=True ignored)

import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------
# SETTINGS
# ----------------------------------------------------------------
SEED = 42
EPOCHS = 15      # max epochs; early stopping may stop sooner
PATIENCE = 3     # stop if combined score doesn't improve for this many epochs

# Dropout must be between 0 and 1 (fraction of units dropped).
# Original list [1.5, 2.0, 2.5, 3.0, 3.5, 3.5, 4.0, 4.5] divided by 10,
# duplicate 3.5 removed.
DROPOUT_RATES = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

HP = {
    "learning_rate": 0.0005,
    "vocab_size": 6000,
    "embedding_dim": 64,
    "rnn_units": 64,
    "max_len": 60,
}

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ==================================================================
# DATA
# ==================================================================
def load_data():
    dataset = load_dataset("zeroshot/twitter-financial-news-sentiment")
    texts = {s: [x["text"].strip() for x in dataset[s] if x["text"].strip()] for s in ["train", "validation"]}
    labels = {s: [int(x["label"]) for x in dataset[s] if x["text"].strip()] for s in ["train", "validation"]}
    return texts["train"], labels["train"], texts["validation"], labels["validation"]


def to_tf_dataset(texts, labels, vectorizer, shuffle=True, batch_size=32):
    ds = tf.data.Dataset.from_tensor_slices((texts, labels))
    if shuffle:
        ds = ds.shuffle(len(texts), seed=SEED)
    return ds.batch(batch_size).map(lambda x, y: (vectorizer(x), y)).prefetch(tf.data.AUTOTUNE)


# ==================================================================
# MODEL — BiSimpleRNN with adjustable dropout
# ==================================================================
def build_model(dropout_rate):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(HP["vocab_size"], HP["embedding_dim"]),
        tf.keras.layers.Bidirectional(tf.keras.layers.SimpleRNN(HP["rnn_units"])),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(3, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(HP["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ==================================================================
# EVALUATE: accuracy + macro-F1 -> combined score
# ==================================================================
def combined_score(model, dataset):
    y_true, y_pred = [], []
    for x, y in dataset:
        y_pred.extend(model.predict(x, verbose=0).argmax(axis=1))
        y_true.extend(y.numpy())
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    return acc, f1, (acc + f1) / 2


# ==================================================================
# RUN ONE EXPERIMENT: train BiSimpleRNN for EPOCHS with given dropout
# ==================================================================
def run_experiment(dropout_rate, train_texts, train_labels, test_texts, test_labels, vectorizer):
    tr_x, val_x, tr_y, val_y = train_test_split(
        train_texts, train_labels, test_size=0.15, stratify=train_labels, random_state=SEED
    )
    train_ds = to_tf_dataset(tr_x, tr_y, vectorizer)
    val_ds = to_tf_dataset(val_x, val_y, vectorizer, shuffle=False)
    test_ds = to_tf_dataset(test_texts, test_labels, vectorizer, shuffle=False)

    print(f"\n--- BiSimpleRNN | dropout={dropout_rate} ---")
    model = build_model(dropout_rate)

    val_history = []
    best_score = -1
    best_weights = None
    epochs_without_improvement = 0
    epochs_trained = 0

    for epoch in range(1, EPOCHS + 1):
        model.fit(train_ds, epochs=1, verbose=0, shuffle=False)
        epochs_trained = epoch
        acc, f1, score = combined_score(model, val_ds)
        val_history.append(score)
        print(f"  Epoch {epoch}: val_acc={acc:.4f} val_f1={f1:.4f} combined={score:.4f}")

        if score > best_score:
            best_score = score
            best_weights = model.get_weights()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print(f"  Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
            break

    if best_weights is not None:
        model.set_weights(best_weights)

    test_acc, test_f1, test_score = combined_score(model, test_ds)
    print(f"Final (stopped epoch {epochs_trained}) | test_acc={test_acc:.4f} "
          f"test_f1={test_f1:.4f} combined={test_score:.4f}")

    return {
        "dropout_rate": dropout_rate,
        "val_history": val_history,
        "test_acc": test_acc,
        "test_f1": test_f1,
        "test_score": test_score,
        "epochs_trained": epochs_trained,
    }


# ==================================================================
# PLOT: compare final combined score across dropout rates
# ==================================================================
def plot_results(results, filename="bisimplernn_dropout_comparison_twitter.png"):
    # Resolve the path relative to this script's own folder, not the
    # current working directory the command happens to be run from.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, filename)

    dropout_rates = [r["dropout_rate"] for r in results]
    scores = [r["test_score"] for r in results]

    plt.figure(figsize=(9, 5))
    plt.plot(dropout_rates, scores, marker="o", color="#2a78d6")
    for x, y in zip(dropout_rates, scores):
        plt.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center")

    plt.xlabel("Dropout Rate")
    plt.ylabel("(Accuracy + Macro-F1) / 2  —  Test Set")
    plt.title(f"BiSimpleRNN — Dropout Comparison (fixed {EPOCHS} epochs)\nTwitter Finance Dataset")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved: {save_path}")


# ==================================================================
# MAIN
# ==================================================================
if __name__ == "__main__":
    train_texts, train_labels, test_texts, test_labels = load_data()

    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=HP["vocab_size"], output_sequence_length=HP["max_len"],
        standardize="lower_and_strip_punctuation"
    )
    vectorizer.adapt(train_texts)

    results = [
        run_experiment(rate, train_texts, train_labels, test_texts, test_labels, vectorizer)
        for rate in DROPOUT_RATES
    ]

    plot_results(results)

    print("\nSUMMARY")
    for r in results:
        print(f"dropout={r['dropout_rate']:<5} test_acc={r['test_acc']:.4f} "
              f"test_f1={r['test_f1']:.4f} combined={r['test_score']:.4f} "
              f"epochs_trained={r['epochs_trained']}")

    best = max(results, key=lambda r: r["test_score"])
    print(f"\nBest dropout rate: {best['dropout_rate']} (combined={best['test_score']:.4f})")
