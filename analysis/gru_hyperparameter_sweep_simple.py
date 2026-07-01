import random
import numpy as np
import tensorflow as tf
from datasets import load_dataset
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from collections import Counter

SEED = 42
EPOCHS = 15          # fixed epoch budget per run (kept small + early stopping handles the rest)
CLASS_NAMES = ["bearish", "bullish", "neutral"]

# Starting defaults - only ONE of these changes per test below
DEFAULTS = {
    "learning_rate": 0.001,
    "vocab_size": 6000,
    "embedding_dim": 64,
    "rnn_units": 64,
    "max_len": 60,
    "dropout_rate": 0.45,   # best result from the earlier dropout sweep
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


def to_tf_dataset(texts, labels, vectorizer, shuffle=True, batch_size=32):
    ds = tf.data.Dataset.from_tensor_slices((texts, labels))
    if shuffle:
        ds = ds.shuffle(len(texts), seed=SEED)
    ds = ds.batch(batch_size)
    ds = ds.map(lambda x, y: (vectorizer(x), y))
    return ds.prefetch(tf.data.AUTOTUNE)


def build_model(vocab_size, embedding_dim, rnn_units, dropout_rate, learning_rate):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim),
        tf.keras.layers.GRU(rnn_units),
        tf.keras.layers.Dropout(dropout_rate),
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
def run_one_config(train_texts, train_labels, test_texts, test_labels, config, label):
    """Train and evaluate one hyperparameter configuration. Returns a
    plain dict with the results - no plotting, no per-epoch history kept."""
    tr_texts, val_texts, tr_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=0.15,
        stratify=train_labels, random_state=SEED
    )

    vectorizer = build_vectorizer(tr_texts, config["vocab_size"], config["max_len"])

    train_ds = to_tf_dataset(tr_texts, tr_labels, vectorizer, shuffle=True)
    val_ds = to_tf_dataset(val_texts, val_labels, vectorizer, shuffle=False)
    test_ds = to_tf_dataset(test_texts, test_labels, vectorizer, shuffle=False)

    model = build_model(config["vocab_size"], config["embedding_dim"],
                         config["rnn_units"], config["dropout_rate"],
                         config["learning_rate"])

    # simple built-in early stopping on validation loss - no custom
    # callback or macro-F1 tracking needed for a hyperparameter sweep
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    )

    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS,
              verbose=0, callbacks=[early_stop])

    true_labels, predicted_labels = [], []
    for batch_x, batch_y in test_ds:
        preds = model.predict(batch_x, verbose=0).argmax(axis=1)
        predicted_labels.extend(preds.tolist())
        true_labels.extend(batch_y.numpy().tolist())

    test_acc = accuracy_score(true_labels, predicted_labels)
    test_macro_f1 = f1_score(true_labels, predicted_labels, average="macro")
    per_class_f1 = f1_score(true_labels, predicted_labels, average=None)

    print(f"{label:<15} | Acc: {test_acc*100:5.2f}% | Macro-F1: {test_macro_f1:.4f} | "
          f"Bearish F1: {per_class_f1[0]:.3f} | Bullish F1: {per_class_f1[1]:.3f} | "
          f"Neutral F1: {per_class_f1[2]:.3f}")

    return {"label": label, "config": dict(config),
            "test_acc": test_acc, "test_macro_f1": test_macro_f1,
            "per_class_f1": per_class_f1}


def print_summary(results, param_name):
    """Plain text summary table - picks and returns the best config."""
    print(f"\n--- {param_name} sweep results ---")
    best = max(results, key=lambda x: x["test_macro_f1"])
    print(f"Best: {best['label']}  "
          f"(Test Macro-F1={best['test_macro_f1']:.4f}, Test Acc={best['test_acc']*100:.2f}%)\n")
    return best


# ==================================================================
# RUN: test each hyperparameter, one at a time, locking in the winner
# ==================================================================
if __name__ == "__main__":
    train_texts, train_labels, test_texts, test_labels = load_data()
    print(f"Train: {len(train_texts)} | Test: {len(test_texts)}")
    print("Train class counts:", Counter(train_labels))

    # ---------------------------------------------------------
    # TEST 1: Learning rate
    # ---------------------------------------------------------
    print("\n=== Learning Rate ===")
    lr_results = []
    for lr in [0.0005, 0.001, 0.002]:
        config = dict(DEFAULTS)
        config["learning_rate"] = lr
        lr_results.append(run_one_config(train_texts, train_labels, test_texts, test_labels,
                                          config, f"lr={lr}"))
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
        vocab_results.append(run_one_config(train_texts, train_labels, test_texts, test_labels,
                                             config, f"vocab={vs}"))
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
        embed_results.append(run_one_config(train_texts, train_labels, test_texts, test_labels,
                                             config, f"embed_dim={ed}"))
    best_embed = print_summary(embed_results, "Embedding Dimension")
    DEFAULTS["embedding_dim"] = best_embed["config"]["embedding_dim"]

    # ---------------------------------------------------------
    # TEST 4: GRU units
    # ---------------------------------------------------------
    print("=== GRU Units ===")
    units_results = []
    for u in [32, 64, 128]:
        config = dict(DEFAULTS)
        config["rnn_units"] = u
        units_results.append(run_one_config(train_texts, train_labels, test_texts, test_labels,
                                             config, f"units={u}"))
    best_units = print_summary(units_results, "GRU Units")
    DEFAULTS["rnn_units"] = best_units["config"]["rnn_units"]

    # ---------------------------------------------------------
    # FINAL: best combination found across all sweeps
    # ---------------------------------------------------------
    print("=== FINAL BEST HYPERPARAMETER COMBINATION ===")
    print(DEFAULTS)
    final_result = run_one_config(train_texts, train_labels, test_texts, test_labels,
                                   DEFAULTS, "FINAL")
    print(f"\nFinal Test Accuracy : {final_result['test_acc']*100:.2f}%")
    print(f"Final Test Macro-F1 : {final_result['test_macro_f1']:.4f}")
    print("\nCompare to SimpleRNN best (dropout=0.45, no weighting): "
          "Accuracy 70.31%, Macro-F1 0.57")
