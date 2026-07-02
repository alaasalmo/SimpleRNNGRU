#  BiSimpleRNN — Kaggle Financial Sentiment (all-data.csv)  (FINAL BEST CONFIG)
#
#  Settings below are the winners from the four-experiment sweep:
#    FINAL OVERALL SUMMARY
#      BiSimpleRNN     best_macro_f1=0.6353  best_epoch=6
#      Best dropout setting (Exp 1): dropout=0.45
#      Best class-weight setting:    OFF
#      Best augmentation setting:    ON
#
#  NOTE: "augmentation ON" in the experiments used WordNet synonym
#  replacement, not nlpaug's four-technique mix (synonym swap, delete,
#  position swap, keyboard typo) — so this script uses WordNet instead,
#  to match what was actually measured. nlpaug is no longer needed.
#
#  NOTE 2: the original script trained using `validation_data=test_ds`,
#  which means the test set was being used to pick when to stop
#  training — that's a data leak (the "test" set stops being a true
#  held-out set once it influences training decisions). This version
#  carves out a separate validation split from the training data for
#  early stopping, and only touches the test set once, at the very end.
#
#  Training uses early stopping on validation macro-F1 (instead of a
#  fixed epoch count) so it naturally lands on the best epoch, whatever
#  it turns out to be for this run (previously epoch 6).

import csv
import random
import numpy as np
import tensorflow as tf
import nltk
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

# CONFIG
DATA_PATH          = "all-data.csv"
SEED                = 42
BATCH_SIZE          = 64
VOCAB_SIZE          = 5000
MAX_LEN             = 50
MAX_EPOCHS          = 30     # early stopping decides the real stopping point
PATIENCE            = 5
DROPOUT_RATE        = 0.45   # winner from the dropout experiment (Exp 1)
USE_CLASS_WEIGHTS   = False  # winner from the class-weight experiment (Exp 2)
USE_AUGMENTATION    = True   # winner from the augmentation experiment (Exp 3)
CLASS_NAMES         = ["negative", "neutral", "positive"]
LABEL_MAP           = {"negative": 0, "neutral": 1, "positive": 2}

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ==================================================================
# STEP 1: Read all-data.csv and split into train / test
# ==================================================================
texts, labels = [], []
with open(DATA_PATH, encoding="latin-1") as f:
    for row in csv.reader(f):
        if len(row) >= 2:
            labels.append(LABEL_MAP.get(row[0].strip().lower(), 1))
            texts.append(row[1].strip())

train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels, test_size=0.2, stratify=labels, random_state=SEED
)

print(f"Total rows   : {len(texts)}")
print(f"Train rows   : {len(train_texts)}  |  Test rows: {len(test_texts)}")

# Carve out a validation split for early stopping (macro-F1 tracking)
tr_texts, val_texts, tr_labels, val_labels = train_test_split(
    train_texts, train_labels, test_size=0.15,
    stratify=train_labels, random_state=SEED
)


# ==================================================================
# STEP 2: NLP-based augmentation (WordNet synonym replacement) — ON
# ==================================================================
nltk.download("wordnet", quiet=True)
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
    """Replaces one random word in `text` with a WordNet synonym."""
    words = text.split()
    if len(words) < 3:
        return text

    indices = list(range(len(words)))
    seed_rng.shuffle(indices)
    for idx in indices:
        synonyms = get_synonyms(words[idx].lower())
        if synonyms:
            words[idx] = seed_rng.choice(synonyms)
            return " ".join(words)
    return text


def augment_texts(texts, labels, seed=SEED):
    """Doubles the training set with one WordNet-augmented copy per example."""
    rng = random.Random(seed)
    augmented_texts, augmented_labels = [], []
    for text, label in zip(texts, labels):
        aug_text = augment_text(text, rng)
        if aug_text:
            augmented_texts.append(aug_text)
            augmented_labels.append(label)
    return list(texts) + augmented_texts, list(labels) + augmented_labels


if USE_AUGMENTATION:
    tr_texts, tr_labels = augment_texts(tr_texts, tr_labels)
    print(f"Augmentation ON  -> training set size: {len(tr_texts)}")
else:
    print("Augmentation OFF")


# ==================================================================
# STEP 3: tf.data pipelines
# ==================================================================
def make_dataset(text_list, label_list, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices((text_list, label_list))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(text_list), seed=SEED)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

train_ds = make_dataset(tr_texts, tr_labels, shuffle=True)
val_ds = make_dataset(val_texts, val_labels, shuffle=False)
test_ds = make_dataset(test_texts, test_labels, shuffle=False)

print("tf.data pipelines ready")


# ==================================================================
# STEP 4: TextVectorization
# ==================================================================
vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode="int",
    output_sequence_length=MAX_LEN
)
vectorizer.adapt(tr_texts)
print(f"Vocabulary built: {len(vectorizer.get_vocabulary())} tokens")

def vectorize(text, label):
    return vectorizer(text), label

train_ds = train_ds.map(vectorize)
val_ds = val_ds.map(vectorize)
test_ds = test_ds.map(vectorize)


# ==================================================================
# STEP 5: Build the Bidirectional SimpleRNN model (winning architecture)
#
#   Embedding       →  word ID → 32-number dense vector
#   Bidirectional   →  wraps SimpleRNN to run the sequence in BOTH directions
#                      Both outputs are concatenated → 64 values instead of 32
#   Dropout         →  0.45 (winner from Exp 1), reduces overfitting
#   Dense           →  outputs probability for each of the 3 classes
# ==================================================================
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=VOCAB_SIZE, output_dim=32, input_length=MAX_LEN),
    tf.keras.layers.Bidirectional(tf.keras.layers.SimpleRNN(32)),
    tf.keras.layers.Dropout(DROPOUT_RATE),
    tf.keras.layers.Dense(3, activation="softmax")
], name="BiSimpleRNN")

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
model.summary()


# ==================================================================
# STEP 6: Train with early stopping on validation macro-F1
#          (class weights OFF, per Exp 2 results)
# ==================================================================
def evaluate_macro_f1(model, dataset, true_labels):
    predicted_labels = model.predict(dataset, verbose=0).argmax(axis=1)
    return f1_score(true_labels, predicted_labels, average="macro")


def train_with_early_stopping(model, train_ds, val_ds, val_labels, max_epochs, patience):
    """
    Trains one epoch at a time, tracks validation macro-F1, keeps the
    best-performing weights, and stops early if there's no improvement
    for `patience` epochs in a row. Returns (best_f1, best_epoch, epochs_trained).
    """
    best_f1 = -1
    best_epoch = 0
    best_weights = None
    epochs_without_improvement = 0
    epochs_trained = 0

    for epoch in range(max_epochs):
        model.fit(train_ds, epochs=1, verbose=0)
        epochs_trained = epoch + 1

        current_f1 = evaluate_macro_f1(model, val_ds, val_labels)

        if current_f1 > best_f1:
            best_f1 = current_f1
            best_epoch = epochs_trained
            best_weights = model.get_weights()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(f"  epoch {epochs_trained:>2} | val_macro_f1={current_f1:.4f} "
              f"{'<-- best' if epochs_without_improvement == 0 else ''}")

        if epochs_without_improvement >= patience:
            print(f"  No improvement for {patience} epochs. Stopping.")
            break

    if best_weights is not None:
        model.set_weights(best_weights)

    return best_f1, best_epoch, epochs_trained


print(f"\nTraining BiSimpleRNN (dropout={DROPOUT_RATE}, "
      f"class_weights={'ON' if USE_CLASS_WEIGHTS else 'OFF'}, "
      f"augmentation={'ON' if USE_AUGMENTATION else 'OFF'})...")
best_f1, best_epoch, epochs_trained = train_with_early_stopping(
    model, train_ds, val_ds, val_labels, max_epochs=MAX_EPOCHS, patience=PATIENCE
)
print(f"\nBest val_macro_f1={best_f1:.4f} at epoch {best_epoch} "
      f"({epochs_trained} epochs trained)")


# ==================================================================
# STEP 7: Evaluate on the held-out test set (never used during training)
# ==================================================================
print("\nEvaluating...")
predicted_labels = model.predict(test_ds, verbose=0).argmax(axis=1)

print("\n" + "─" * 52)
print("CLASSIFICATION REPORT  (BiSimpleRNN + WordNet augmentation)")
print("─" * 52)
print(classification_report(test_labels, predicted_labels, target_names=CLASS_NAMES, zero_division=0))
print(f"Test macro-F1  : {f1_score(test_labels, predicted_labels, average='macro'):.4f}")
print(f"Best val epoch : {best_epoch} (best val_macro_f1={best_f1:.4f})")


# ==================================================================
# STEP 8: Predict any sentence
# ==================================================================
def predict(sentence):
    vec = vectorizer([sentence])
    probs = model.predict(vec, verbose=0)[0]
    idx = probs.argmax()
    label = CLASS_NAMES[idx]
    conf = probs[idx] * 100
    print(f"  Text  : {sentence[:70]}")
    print(f"  Result: {label.upper()}  ({conf:.0f}% confident)\n")

print("─" * 52)
print("SAMPLE PREDICTIONS")
print("─" * 52)
predict("The company reported record profits and strong revenue growth")
predict("Massive layoffs announced as sales decline sharply")
predict("The firm maintained normal operations this quarter")
