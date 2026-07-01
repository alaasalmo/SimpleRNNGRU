import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import nltk
from datasets import load_dataset
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

HP = {
    "learning_rate": 0.0005,
    "vocab_size": 6000,
    "embedding_dim": 32,
    "rnn_units": 32,
    "max_len": 60,
    "dropout_rate": 0.45,   # default / baseline dropout
}

DROPOUT_SETTINGS_TO_TEST = [0.0, 0.45]   # experiment 1: no dropout vs dropout=0.6

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ==================================================================
# PART 1: LOAD DATA
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
def evaluate_macro_f1(model, dataset):
    """Runs the model over a tf.data dataset and returns macro-F1."""
    true_labels, predicted_labels = [], []
    for batch_x, batch_y in dataset:
        predictions = model.predict(batch_x, verbose=0).argmax(axis=1)
        predicted_labels.extend(predictions.tolist())
        true_labels.extend(batch_y.numpy().tolist())
    return f1_score(true_labels, predicted_labels, average="macro")


def train_with_tracking(model, train_ds, val_ds, class_weight_dict, max_epochs, patience):
    """
    Trains one epoch at a time, checks macro-F1 on the validation set
    after each epoch, keeps the best-scoring weights, and stops early
    if there's no improvement for `patience` epochs in a row.
    Returns a plain dict: f1_per_epoch, best_f1, best_epoch.
    """
    f1_per_epoch = []
    best_f1 = -1
    best_epoch = 0
    best_weights = None
    epochs_without_improvement = 0

    for epoch in range(max_epochs):
        model.fit(train_ds, epochs=1, verbose=0, class_weight=class_weight_dict)

        current_f1 = evaluate_macro_f1(model, val_ds)
        f1_per_epoch.append(current_f1)
        print(f"  Epoch {epoch + 1}: val_macro_f1 = {current_f1:.4f}")

        if current_f1 > best_f1:
            best_f1 = current_f1
            best_epoch = epoch + 1
            best_weights = model.get_weights()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"  No improvement for {patience} epochs. Stopping.")
            break

    if best_weights is not None:
        model.set_weights(best_weights)

    return {"f1_per_epoch": f1_per_epoch, "best_f1": best_f1, "best_epoch": best_epoch}


# ==================================================================
# PART 5: CLASS WEIGHTS
# ==================================================================
def get_class_weights(labels):
    classes = np.unique(labels)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    return {int(c): float(w) for c, w in zip(classes, weights)}


# ==================================================================
# PART 5b: NLP-BASED TEXT AUGMENTATION (WordNet synonym replacement)
# ==================================================================
# Synonyms are looked up automatically with NLTK's WordNet — a large
# lexical database of English word relationships — instead of a
# hand-picked synonym list or purely mechanical word shuffling.
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet", quiet=True)
try:
    nltk.data.find("corpora/omw-1.4")
except LookupError:
    nltk.download("omw-1.4", quiet=True)

from nltk.corpus import wordnet


def get_synonyms(word):
    """Look up single-word WordNet synonyms for `word` (case-insensitive)."""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            name = lemma.name().replace("_", " ")
            if " " in name:
                continue  # skip multi-word phrases, keep augmented text natural
            if name.lower() != word.lower():
                synonyms.add(name)
    return list(synonyms)


def augment_text(text, seed_rng):
    """
    Builds one augmented version of `text` by replacing a random word
    with a WordNet synonym. Tries words in random order until one has
    a usable synonym; returns the original text if none is found.
    """
    words = text.split()
    if len(words) < 3:
        return text  # too short to safely augment

    indices = list(range(len(words)))
    seed_rng.shuffle(indices)
    for idx in indices:
        synonyms = get_synonyms(words[idx].lower())
        if synonyms:
            words[idx] = seed_rng.choice(synonyms)
            return " ".join(words)

    return text  # no swappable word found


def augment_texts(texts, labels, seed=SEED):
    """
    For every training example, generates one WordNet-augmented copy
    and appends it to the dataset. Doubles the size of the training set.
    """
    rng = random.Random(seed)
    augmented_texts, augmented_labels = [], []

    for text, label in zip(texts, labels):
        words = text.split()
        if not words:
            continue
        aug_text = augment_text(text, rng)
        if aug_text:
            augmented_texts.append(aug_text)
            augmented_labels.append(label)

    combined_texts = list(texts) + augmented_texts
    combined_labels = list(labels) + augmented_labels
    return combined_texts, combined_labels


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

    class_weight_dict = get_class_weights(tr_labels) if use_class_weights else None
    if class_weight_dict:
        print(f"Class weights: {class_weight_dict}")

    print(f"\n--- {run_name} ---")
    print(f"dropout={dropout_rate}, class_weights={'ON' if use_class_weights else 'OFF'}, "
          f"bidirectional={bidirectional} (up to {MAX_EPOCHS} epochs, patience={PATIENCE})")

    result = train_with_tracking(model, train_ds, val_ds, class_weight_dict,
                                  max_epochs=MAX_EPOCHS, patience=PATIENCE)

    true_labels, predicted_labels = [], []
    for batch_x, batch_y in test_ds:
        predictions = model.predict(batch_x, verbose=0).argmax(axis=1)
        predicted_labels.extend(predictions.tolist())
        true_labels.extend(batch_y.numpy().tolist())

    print(f"Best epoch: {result['best_epoch']} (out of {len(result['f1_per_epoch'])} trained)")
    print(classification_report(true_labels, predicted_labels,
                                 target_names=CLASS_NAMES, zero_division=0))

    return result


# ==================================================================
# PART 7: PLOTTING (shared helper)
# ==================================================================
def plot_comparison(results, title, filename):
    """results: list of (label, result_dict) tuples."""
    plt.figure(figsize=(9, 5))
    colors = ["#1baf7a", "#2a78d6", "#e34948", "#a26fd4"]

    for (label, result), color in zip(results, colors):
        f1_per_epoch = result["f1_per_epoch"]
        best_epoch = result["best_epoch"]
        epochs = range(1, len(f1_per_epoch) + 1)
        plt.plot(epochs, f1_per_epoch, label=label, color=color)
        plt.scatter(best_epoch, f1_per_epoch[best_epoch - 1],
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
    print(f"\n{'='*45}\nEXPERIMENT 1: DROPOUT 0.45 vs NO DROPOUT\n{'='*45}")
    dropout_results = []
    for rate in DROPOUT_SETTINGS_TO_TEST:
        label = "no dropout" if rate == 0.0 else f"dropout={rate}"
        result = train_and_evaluate(
            train_texts, train_labels, test_texts, test_labels, vectorizer,
            dropout_rate=rate, use_class_weights=False, bidirectional=False,
            run_name=label
        )
        dropout_results.append((label, result))

    plot_comparison(
        dropout_results,
        title="Experiment 1: Macro-F1 — Dropout 0.45 vs No Dropout\n(stars = each curve's best epoch)",
        filename="dropout_comparison_simplernn_twitter.png"
    )

    print("\nEXPERIMENT 1 SUMMARY")
    for label, result in dropout_results:
        print(f"  {label:<15} best_macro_f1={result['best_f1']:.4f}  best_epoch={result['best_epoch']}")

    best_label, best_result = max(dropout_results, key=lambda x: x[1]['best_f1'])
    best_dropout = 0.0 if best_label == "no dropout" else float(best_label.split("=")[1])
    print(f"\nBest setting: {best_label} (macro-F1={best_result['best_f1']:.4f})")

    # ============================================================
    # EXPERIMENT 2: Class weights ON vs OFF (using best dropout)
    # ============================================================
    print(f"\n{'='*45}\nEXPERIMENT 2: CLASS WEIGHTS (dropout fixed at {best_dropout})\n{'='*45}")

    result_no_weight = train_and_evaluate(
        train_texts, train_labels, test_texts, test_labels, vectorizer,
        dropout_rate=best_dropout, use_class_weights=False, bidirectional=False,
        run_name="Class weights OFF"
    )
    result_with_weight = train_and_evaluate(
        train_texts, train_labels, test_texts, test_labels, vectorizer,
        dropout_rate=best_dropout, use_class_weights=True, bidirectional=False,
        run_name="Class weights ON"
    )

    weight_results = [
        ("No class weights", result_no_weight),
        ("With class weights", result_with_weight),
    ]
    plot_comparison(
        weight_results,
        title=f"Experiment 2: Macro-F1 With vs Without Class Weights\n(dropout={best_dropout}, stars = best epoch)",
        filename="class_weight_comparison_simplernn_twitter.png"
    )

    print("\nEXPERIMENT 2 SUMMARY")
    for label, result in weight_results:
        print(f"  {label:<20} best_macro_f1={result['best_f1']:.4f}  best_epoch={result['best_epoch']}")

    # pick best class-weight setting to carry forward
    use_weights_for_exp3 = result_with_weight['best_f1'] > result_no_weight['best_f1']
    print(f"\nUsing class_weights={'ON' if use_weights_for_exp3 else 'OFF'} for Experiment 3 "
          f"(whichever scored higher above)")

    # ============================================================
    # EXPERIMENT 3: Augmentation ON vs OFF (using best dropout + class weights)
    # ============================================================
    print(f"\n{'='*45}\nEXPERIMENT 3: AUGMENTATION (dropout fixed at {best_dropout}, "
          f"class_weights={'ON' if use_weights_for_exp3 else 'OFF'})\n{'='*45}")

    result_no_aug = train_and_evaluate(
        train_texts, train_labels, test_texts, test_labels, vectorizer,
        dropout_rate=best_dropout, use_class_weights=use_weights_for_exp3,
        bidirectional=False, run_name="Augmentation OFF"
    )

    aug_train_texts, aug_train_labels = augment_texts(train_texts, train_labels)
    print(f"Augmented training set size: {len(train_texts)} -> {len(aug_train_texts)}")

    result_with_aug = train_and_evaluate(
        aug_train_texts, aug_train_labels, test_texts, test_labels, vectorizer,
        dropout_rate=best_dropout, use_class_weights=use_weights_for_exp3,
        bidirectional=False, run_name="Augmentation ON"
    )

    aug_results = [
        ("No augmentation", result_no_aug),
        ("With augmentation", result_with_aug),
    ]
    plot_comparison(
        aug_results,
        title=f"Experiment 3: Macro-F1 With vs Without Augmentation\n"
              f"(dropout={best_dropout}, class_weights={'ON' if use_weights_for_exp3 else 'OFF'}, stars = best epoch)",
        filename="augmentation_comparison_simplernn_twitter.png"
    )

    print("\nEXPERIMENT 3 SUMMARY")
    for label, result in aug_results:
        print(f"  {label:<20} best_macro_f1={result['best_f1']:.4f}  best_epoch={result['best_epoch']}")

    # pick best augmentation setting to carry into experiment 4
    use_aug_for_exp4 = result_with_aug['best_f1'] > result_no_aug['best_f1']
    exp4_train_texts = aug_train_texts if use_aug_for_exp4 else train_texts
    exp4_train_labels = aug_train_labels if use_aug_for_exp4 else train_labels
    print(f"\nUsing augmentation={'ON' if use_aug_for_exp4 else 'OFF'} for Experiment 4 "
          f"(whichever scored higher above)")

    # ============================================================
    # EXPERIMENT 4: SimpleRNN vs Bidirectional SimpleRNN
    # (best dropout + best class-weight + best augmentation setting from above)
    # ============================================================
    print(f"\n{'='*45}\nEXPERIMENT 4: SimpleRNN vs BiSimpleRNN\n{'='*45}")

    result_simple = train_and_evaluate(
        exp4_train_texts, exp4_train_labels, test_texts, test_labels, vectorizer,
        dropout_rate=best_dropout, use_class_weights=use_weights_for_exp3,
        bidirectional=False, run_name="SimpleRNN"
    )
    result_bi = train_and_evaluate(
        exp4_train_texts, exp4_train_labels, test_texts, test_labels, vectorizer,
        dropout_rate=best_dropout, use_class_weights=use_weights_for_exp3,
        bidirectional=True, run_name="Bidirectional SimpleRNN"
    )

    arch_results = [
        ("SimpleRNN", result_simple),
        ("BiSimpleRNN", result_bi),
    ]
    plot_comparison(
        arch_results,
        title=f"Experiment 4: SimpleRNN vs BiSimpleRNN\n(dropout={best_dropout}, "
              f"class_weights={'ON' if use_weights_for_exp3 else 'OFF'}, "
              f"augmentation={'ON' if use_aug_for_exp4 else 'OFF'}, stars = best epoch)",
        filename="simplernn_vs_bisimplernn_comparison_simplernn_twitter.png"
    )

    print("\nEXPERIMENT 4 SUMMARY")
    for label, result in arch_results:
        print(f"  {label:<15} best_macro_f1={result['best_f1']:.4f}  best_epoch={result['best_epoch']}")

    # ============================================================
    # FINAL OVERALL SUMMARY
    # ============================================================
    print(f"\n{'='*45}\nFINAL OVERALL SUMMARY\n{'='*45}")
    print(f"Best dropout setting (Exp 1):     {best_label}")
    print(f"Best class-weight setting:        {'ON' if use_weights_for_exp3 else 'OFF'}")
    print(f"Best augmentation setting (Exp 3): {'ON' if use_aug_for_exp4 else 'OFF'}")
    print(f"SimpleRNN best macro-F1:     {result_simple['best_f1']:.4f} (epoch {result_simple['best_epoch']})")
    print(f"BiSimpleRNN best macro-F1:   {result_bi['best_f1']:.4f} (epoch {result_bi['best_epoch']})")
