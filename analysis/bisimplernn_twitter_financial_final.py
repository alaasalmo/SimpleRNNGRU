#  BiSimpleRNN — Financial Sentiment Analysis (FINAL BEST CONFIG)
#  Dataset: zeroshot/twitter-financial-news-sentiment
#
#  Settings below are the winners from the four-experiment sweep:
#    EXPERIMENT 4 SUMMARY
#      BiSimpleRNN     best_macro_f1=0.9237  best_epoch=13
#    FINAL OVERALL SUMMARY
#      Best dropout setting (Exp 1):     dropout=0.45
#      Best class-weight setting:        OFF
#      Best augmentation setting (Exp 3): ON
#      SimpleRNN best macro-F1:     0.8417 (epoch 20)
#      BiSimpleRNN best macro-F1:   0.9237 (epoch 13)
#
#  NOTE: "augmentation ON" in the experiments above used WordNet
#  synonym replacement, not nlpaug's four-technique mix (synonym swap,
#  delete, position swap, keyboard typo). This script uses the same
#  WordNet approach so the config here matches what was actually
#  measured — nlpaug is no longer needed.
#
#  Training uses early stopping on validation macro-F1 (instead of a
#  fixed epoch count) so it naturally lands on the best epoch, whatever
#  it turns out to be for this run (previously epoch 13).

import random, time
import numpy as np
import tensorflow as tf
import nltk
from datasets import load_dataset
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# CONFIG
SEED               = 42
BATCH_SIZE         = 32
VOCAB_SIZE         = 6000
MAX_LEN            = 60
MAX_EPOCHS         = 30     # early stopping decides the real stopping point
PATIENCE           = 5
DROPOUT_RATE       = 0.45   # winner from the dropout experiment (Exp 1)
USE_CLASS_WEIGHTS  = False  # winner from the class-weight experiment (Exp 2)
USE_AUGMENTATION   = True   # winner from the augmentation experiment (Exp 3)
CLASS_NAMES        = ["bearish", "bullish", "neutral"]

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ==================================================================
# STEP 1: Load dataset
# ==================================================================
dataset = load_dataset("zeroshot/twitter-financial-news-sentiment")

def extract(split):
    texts, labels = [], []
    for item in dataset[split]:
        text = item.get("text", "").strip()
        lbl = int(item.get("label", 0))
        if text:
            texts.append(text)
            labels.append(lbl)
    return texts, labels

train_texts, train_labels = extract("train")
test_texts, test_labels = extract("validation")

print(f"Train: {len(train_texts)} | Test: {len(test_texts)}")

# Carve out a validation split for early stopping (macro-F1 tracking)
tr_texts, val_texts, tr_labels, val_labels = train_test_split(
    train_texts, train_labels, test_size=0.15,
    stratify=train_labels, random_state=SEED
)


# ==================================================================
# STEP 2: NLP-based augmentation (WordNet synonym replacement) — ON
# ==================================================================
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
def make_dataset(texts, labels, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((texts, labels))
    if shuffle:
        ds = ds.shuffle(len(texts), seed=SEED)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

train_ds = make_dataset(tr_texts, tr_labels, shuffle=True)
val_ds = make_dataset(val_texts, val_labels, shuffle=False)
test_ds = make_dataset(test_texts, test_labels, shuffle=False)


# ==================================================================
# STEP 4: TextVectorization
# ==================================================================
vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode="int",
    output_sequence_length=MAX_LEN,
    standardize="lower_and_strip_punctuation"
)
vectorizer.adapt(tr_texts)
print(f"\nVocabulary: {len(vectorizer.get_vocabulary())} tokens")

def vectorize(text, label):
    return vectorizer(text), label

train_ds = train_ds.map(vectorize)
val_ds = val_ds.map(vectorize)
test_ds = test_ds.map(vectorize)


# ==================================================================
# STEP 5: Build BiSimpleRNN model (winning architecture)
# ==================================================================
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(VOCAB_SIZE, 64, input_length=MAX_LEN),
    tf.keras.layers.Bidirectional(tf.keras.layers.SimpleRNN(64)),
    tf.keras.layers.Dropout(DROPOUT_RATE),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(3, activation="softmax")
], name="BiSimpleRNN")

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
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
t0 = time.time()
best_f1, best_epoch, epochs_trained = train_with_early_stopping(
    model, train_ds, val_ds, val_labels, max_epochs=MAX_EPOCHS, patience=PATIENCE
)
train_time = time.time() - t0
print(f"\nDone in {train_time:.0f}s — best val_macro_f1={best_f1:.4f} at epoch {best_epoch} "
      f"({epochs_trained} epochs trained)")


# ==================================================================
# STEP 7: Evaluate on the test set
# ==================================================================
print("\nEvaluating...")
predicted_labels = model.predict(test_ds, verbose=0).argmax(axis=1)

print("\n" + "─" * 52)
print("CLASSIFICATION REPORT  (BiSimpleRNN + WordNet augmentation)")
print("─" * 52)
print(classification_report(test_labels, predicted_labels, target_names=CLASS_NAMES, zero_division=0))
print(f"Test accuracy   : {accuracy_score(test_labels, predicted_labels)*100:.2f}%")
print(f"Test macro-F1   : {f1_score(test_labels, predicted_labels, average='macro'):.4f}")
print(f"Best val epoch  : {best_epoch} (best val_macro_f1={best_f1:.4f})")
print(f"Training time   : {train_time:.0f}s")
print(f"Parameters      : {model.count_params():,}")


# ==================================================================
# STEP 8: Predict any sentence
# ==================================================================
def predict(sentence):
    probs = model.predict(vectorizer([sentence]), verbose=0)[0]
    idx = probs.argmax()
    print(f"  Text  : {sentence[:70]}")
    print(f"  Result: {CLASS_NAMES[idx].upper()}  ({probs[idx]*100:.0f}% confident)")
    print(f"  Scores: bear={probs[0]:.2f}  bull={probs[1]:.2f}  neu={probs[2]:.2f}\n")

print("─" * 52)
print("SAMPLE PREDICTIONS  (BiSimpleRNN + WordNet augmentation)")
print("─" * 52)
predict("Apple stock surges to all-time high on record earnings")
predict("Markets crash as recession fears grip investors worldwide")
predict("Fed holds rates steady as inflation remains uncertain")
predict("Tesla shares drop 15% after missing delivery targets badly")
predict("Goldman Sachs upgrades sector outlook citing strong growth")
