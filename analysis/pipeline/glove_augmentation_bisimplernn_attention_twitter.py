"""
Augmentation -> Pretrained GloVe -> Bidirectional SimpleRNN -> Attention -> Dense -> Softmax
Twitter financial sentiment dataset (zeroshot/twitter-financial-news-sentiment).

Uses the winning hyperparameters found in earlier sweeps:
  - dropout_rate  = 0.10   (best from the dropout sweep)
  - learning_rate = 0.0005 (best from the hyperparameter sweep)
  - vocab_size    = 6000
  - rnn_units     = 64

NOTE on embedding_dim: the earlier sweep found embedding_dim=64 best, but
that was for a *trainable, randomly-initialized* embedding layer. Pretrained
GloVe vectors only come in fixed sizes (50/100/200/300), so embedding_dim=64
isn't an option here. This script uses GloVe's 100-dimensional vectors
(glove.6B.100d.txt) instead -- the closest available size -- and keeps
everything else from the tuned config as-is.

Early stopping monitors the same combined (accuracy + macro-F1) / 2 score
used throughout this project, and reports the best epoch found.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # silence TF C++ INFO/WARNING/ERROR logs

import re
import random
import zipfile
import urllib.request

import numpy as np
import tensorflow as tf
from datasets import load_dataset
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")

import nltk
from nltk.corpus import wordnet

# A small hardcoded stopword list — avoids depending on nltk's separate
# "stopwords" corpus download, which is one more thing that can fail/get
# corrupted on some Windows setups. WordNet (for synonyms) is still needed
# and is handled separately in ensure_nltk_data() below.
STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "if", "is", "are", "was", "were",
    "be", "been", "being", "to", "of", "in", "on", "for", "with", "as",
    "at", "by", "from", "this", "that", "these", "those", "it", "its",
    "i", "you", "he", "she", "we", "they", "them", "his", "her", "their",
    "our", "your", "my", "me", "him", "us", "not", "no", "do", "does",
    "did", "will", "would", "shall", "should", "can", "could", "may",
    "might", "must", "have", "has", "had", "so", "than", "then", "there",
    "here", "up", "down", "out", "over", "under", "again", "further",
    "once", "s", "t", "re", "ve", "ll", "d", "m",
}

# ----------------------------------------------------------------
# SETTINGS
# ----------------------------------------------------------------
SEED = 42
EPOCHS = 30      # max epochs; early stopping decides the real stopping point
PATIENCE = 5     # stop if combined score doesn't improve for this many epochs
CLASS_NAMES = ["bearish", "bullish", "neutral"]

HP = {
    "learning_rate": 0.0005,   # best from hyperparameter sweep
    "vocab_size": 6000,        # best from hyperparameter sweep
    "embedding_dim": 100,      # fixed by GloVe's available sizes (see note above)
    "rnn_units": 64,           # best from hyperparameter sweep
    "dropout_rate": 0.10,      # best from dropout sweep
    "max_len": 60,
}

GLOVE_DIR = "glove"
GLOVE_FILE = os.path.join(GLOVE_DIR, "glove.6B.100d.txt")
GLOVE_ZIP_URL = "https://nlp.stanford.edu/data/glove.6B.zip"

AUGMENT_FRACTION = 0.30   # fraction of training texts to augment (adds a paraphrased copy)
AUGMENT_WORD_PROB = 0.20  # per-word probability of a WordNet synonym swap

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ==================================================================
# NLTK SETUP (WordNet synonyms; stopwords are hardcoded above, no download needed)
# ==================================================================
def ensure_nltk_data():
    """
    Ensures WordNet (and its multilingual companion omw-1.4, which WordNet
    needs internally) are available. Downloads are attempted with visible
    output (quiet=False) so any failure is obvious rather than silently
    breaking later inside augment_text().
    """
    for pkg in ["wordnet", "omw-1.4"]:
        try:
            nltk.data.find(f"corpora/{pkg}")
        except LookupError:
            print(f"Downloading NLTK '{pkg}' corpus...")
            success = nltk.download(pkg, quiet=False)
            if not success:
                raise RuntimeError(
                    f"\nFailed to download NLTK corpus '{pkg}'.\n"
                    f"This is usually a network/proxy/antivirus issue on Windows.\n"
                    f"Try fixing it manually:\n"
                    f"  1. Open a Python shell and run: import nltk; nltk.download()\n"
                    f"     (this opens a GUI - select and download 'wordnet' and 'omw-1.4')\n"
                    f"  2. Or delete the folder %APPDATA%\\nltk_data and re-run this script\n"
                    f"     (a partially-downloaded/corrupted folder is a common cause)\n"
                    f"  3. Or download manually from https://github.com/nltk/nltk_data "
                    f"and place under %APPDATA%\\nltk_data\\corpora\\{pkg}\n"
                )


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


# ==================================================================
# AUGMENTATION: simple WordNet synonym replacement on raw text,
# applied BEFORE tokenization/embedding (not after, unlike a common mistake)
# ==================================================================
def get_synonym(word, stop_words):
    """Return a single WordNet synonym for `word`, or None if unavailable."""
    if word.lower() in stop_words or not word.isalpha():
        return None
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            candidate = lemma.name().replace("_", " ")
            if candidate.lower() != word.lower():
                synonyms.add(candidate)
    if not synonyms:
        return None
    return random.choice(list(synonyms))


def augment_text(text, stop_words, word_prob=AUGMENT_WORD_PROB):
    """Randomly swap some words in `text` for a WordNet synonym."""
    words = text.split()
    new_words = []
    for word in words:
        if random.random() < word_prob:
            synonym = get_synonym(word, stop_words)
            new_words.append(synonym if synonym else word)
        else:
            new_words.append(word)
    return " ".join(new_words)


def augment_dataset(texts, labels, fraction=AUGMENT_FRACTION):
    """
    Adds an augmented (synonym-swapped) copy of a random `fraction` of the
    training set on top of the original data. Returns the combined lists.
    """
    ensure_nltk_data()
    stop_words = STOP_WORDS

    n_to_augment = int(len(texts) * fraction)
    indices = random.sample(range(len(texts)), n_to_augment)

    aug_texts = [augment_text(texts[i], stop_words) for i in indices]
    aug_labels = [labels[i] for i in indices]

    combined_texts = texts + aug_texts
    combined_labels = labels + aug_labels
    print(f"Augmentation: added {len(aug_texts)} synonym-swapped examples "
          f"({fraction*100:.0f}% of training set)")
    return combined_texts, combined_labels


# ==================================================================
# VECTORIZER
# ==================================================================
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


# ==================================================================
# PRETRAINED GLOVE EMBEDDINGS
# ==================================================================
def ensure_glove_file():
    """Downloads and extracts glove.6B.100d.txt if it isn't present already."""
    if os.path.exists(GLOVE_FILE):
        return
    os.makedirs(GLOVE_DIR, exist_ok=True)
    zip_path = os.path.join(GLOVE_DIR, "glove.6B.zip")
    print("GloVe file not found locally — downloading glove.6B.zip (~800MB)...")
    urllib.request.urlretrieve(GLOVE_ZIP_URL, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extract("glove.6B.100d.txt", GLOVE_DIR)
    os.remove(zip_path)
    print(f"GloVe vectors ready at {GLOVE_FILE}")


def load_glove_vectors(glove_path):
    """Loads GloVe vectors from disk into a {word: np.array} dict."""
    embeddings = {}
    with open(glove_path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word = parts[0]
            vector = np.asarray(parts[1:], dtype="float32")
            embeddings[word] = vector
    return embeddings


def build_embedding_matrix(vectorizer, glove_vectors, embedding_dim):
    """Builds a [vocab_size, embedding_dim] matrix aligned to the vectorizer's vocabulary."""
    vocab = vectorizer.get_vocabulary()
    matrix = np.zeros((len(vocab), embedding_dim), dtype="float32")
    hits = 0
    for i, word in enumerate(vocab):
        vector = glove_vectors.get(word)
        if vector is not None:
            matrix[i] = vector
            hits += 1
    print(f"GloVe coverage: {hits}/{len(vocab)} vocabulary words found "
          f"({hits / len(vocab) * 100:.1f}%)")
    return matrix


# ==================================================================
# ATTENTION LAYER (simple additive/Bahdanau-style pooling over BiRNN outputs)
# ==================================================================
class AttentionPooling(tf.keras.layers.Layer):
    """
    Learns a weight for each timestep of the BiRNN output, then returns a
    single weighted-sum vector per sequence (attention pooling).
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        hidden_dim = input_shape[-1]
        self.score_dense = tf.keras.layers.Dense(hidden_dim, activation="tanh")
        self.context_vector = self.add_weight(
            name="context_vector", shape=(hidden_dim, 1),
            initializer="glorot_uniform", trainable=True
        )
        super().build(input_shape)

    def call(self, inputs):
        # inputs: [batch, timesteps, hidden_dim]
        scores = self.score_dense(inputs)                     # [batch, timesteps, hidden_dim]
        scores = tf.matmul(scores, self.context_vector)        # [batch, timesteps, 1]
        weights = tf.nn.softmax(scores, axis=1)                 # [batch, timesteps, 1]
        weighted_sum = tf.reduce_sum(inputs * weights, axis=1)  # [batch, hidden_dim]
        return weighted_sum


# ==================================================================
# MODEL — Augmentation (done earlier, on raw text) -> GloVe -> BiSimpleRNN -> Attention -> Dense -> Softmax
# ==================================================================
def build_model(vocab_size, embedding_dim, rnn_units, dropout_rate, learning_rate, embedding_matrix):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(
            vocab_size, embedding_dim,
            embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
            trainable=False,   # GloVe vectors are frozen; only the rest of the network learns
        ),
        tf.keras.layers.Bidirectional(tf.keras.layers.SimpleRNN(rnn_units, return_sequences=True)),
        AttentionPooling(),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(3, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ==================================================================
# EVALUATE: accuracy + macro-F1 -> combined score
# ==================================================================
def evaluate_combined_score(model, dataset, true_labels):
    predicted_labels = model.predict(dataset, verbose=0).argmax(axis=1)
    acc = accuracy_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels, average="macro")
    return acc, f1, (acc + f1) / 2


# ==================================================================
# TRAIN WITH EARLY STOPPING, finding the best epoch by combined score
# ==================================================================
def train_with_early_stopping(model, train_ds, val_ds, val_labels, max_epochs, patience):
    best_combined = -1
    best_epoch = 0
    best_weights = None
    epochs_without_improvement = 0
    epochs_trained = 0
    val_history = []

    for epoch in range(1, max_epochs + 1):
        model.fit(train_ds, epochs=1, verbose=0, shuffle=False)
        epochs_trained = epoch

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
            print(f"  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    if best_weights is not None:
        model.set_weights(best_weights)

    return best_combined, best_epoch, epochs_trained, val_history


def print_label_distribution(labels, name):
    from collections import Counter
    counts = Counter(labels)
    total = len(labels)
    parts = []
    for idx, class_name in enumerate(CLASS_NAMES):
        count = counts.get(idx, 0)
        pct = (count / total * 100) if total > 0 else 0.0
        parts.append(f"{class_name.capitalize()}: {count} ({pct:.1f}%)")
    print(f"{name} class counts -> " + " | ".join(parts))


# ==================================================================
# MAIN
# ==================================================================
if __name__ == "__main__":
    train_texts, train_labels, test_texts, test_labels = load_data()
    print(f"Train: {len(train_texts)} | Test: {len(test_texts)}")
    print_label_distribution(train_labels, "Train")
    print_label_distribution(test_labels, "Test")

    # Split off a validation set BEFORE augmenting, so augmented examples
    # never leak into validation/test.
    tr_texts, val_texts, tr_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=0.15,
        stratify=train_labels, random_state=SEED
    )

    # Step 1: Augmentation (on raw text, before embedding/vectorization)
    tr_texts, tr_labels = augment_dataset(tr_texts, tr_labels)
    print_label_distribution(tr_labels, "Train split (post-augmentation)")
    print_label_distribution(val_labels, "Val split")

    # Step 2: Vectorizer (built on the augmented training text)
    vectorizer = build_vectorizer(tr_texts, HP["vocab_size"], HP["max_len"])

    train_ds = to_tf_dataset(tr_texts, tr_labels, vectorizer, shuffle=True)
    val_ds = to_tf_dataset(val_texts, val_labels, vectorizer, shuffle=False)
    test_ds = to_tf_dataset(test_texts, test_labels, vectorizer, shuffle=False)

    # Step 3: Pretrained GloVe embeddings
    ensure_glove_file()
    glove_vectors = load_glove_vectors(GLOVE_FILE)
    embedding_matrix = build_embedding_matrix(vectorizer, glove_vectors, HP["embedding_dim"])

    # Step 4: Build Augmentation -> GloVe -> BiSimpleRNN -> Attention -> Dense -> Softmax
    model = build_model(
        vocab_size=HP["vocab_size"], embedding_dim=HP["embedding_dim"],
        rnn_units=HP["rnn_units"], dropout_rate=HP["dropout_rate"],
        learning_rate=HP["learning_rate"], embedding_matrix=embedding_matrix,
    )
    model.summary()

    # Step 5: Train with early stopping, tracking the best epoch
    print("\n=== Training ===")
    best_combined, best_epoch, epochs_trained, val_history = train_with_early_stopping(
        model, train_ds, val_ds, val_labels, max_epochs=EPOCHS, patience=PATIENCE
    )

    # Step 6: Final test evaluation (using best-epoch weights, already restored)
    test_acc, test_f1, test_combined = evaluate_combined_score(model, test_ds, test_labels)
    predicted_labels = model.predict(test_ds, verbose=0).argmax(axis=1)
    per_class_f1 = f1_score(test_labels, predicted_labels, average=None)

    print("\n" + "=" * 60)
    print("FINAL RESULTS — Augmentation -> GloVe -> BiSimpleRNN -> Attention -> Dense -> Softmax")
    print("=" * 60)
    print(f"Hyperparameters used: {HP}")
    print(f"Test Accuracy   : {test_acc*100:.2f}%")
    print(f"Test Macro-F1   : {test_f1:.4f}")
    print(f"Test Combined   : {test_combined:.4f}")
    print(f"Bearish F1: {per_class_f1[0]:.3f} | Bullish F1: {per_class_f1[1]:.3f} | "
          f"Neutral F1: {per_class_f1[2]:.3f}")
    print(f"Best epoch      : {best_epoch} (out of {epochs_trained} trained)")
