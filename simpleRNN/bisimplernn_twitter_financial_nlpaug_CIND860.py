#  BiSimpleRNN — Financial Sentiment Analysis
#  Dataset: zeroshot/twitter-financial-news-sentiment
#  + nlpaug data augmentation

import random, time
import numpy as np
import tensorflow as tf
import nltk
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac
from datasets import load_dataset
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

# ── Download NLTK resources required by nlpaug (runs once) ───────────
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('averaged_perceptron_tagger',     quiet=True)
nltk.download('wordnet',                        quiet=True)
nltk.download('omw-1.4',                        quiet=True)

# CONFIG
SEED        = 42
BATCH_SIZE  = 32
VOCAB_SIZE  = 6000
MAX_LEN     = 60
EPOCHS      = 10
CLASS_NAMES = ["bearish", "bullish", "neutral"]

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# STEP 1: Load dataset
dataset = load_dataset("zeroshot/twitter-financial-news-sentiment")

def extract(split):
    texts, labels = [], []
    for item in dataset[split]:
        text = item.get("text", "").strip()
        lbl  = int(item.get("label", 0))
        if text:
            texts.append(text)
            labels.append(lbl)
    return texts, labels

train_texts, train_labels = extract("train")
test_texts,  test_labels  = extract("validation")

print(f"Train: {len(train_texts)} | Test: {len(test_texts)}")


# STEP 1b: Augmentation with nlpaug
#
#   Four augmenters are applied randomly to every training sentence:
#
#   SynonymAug          →  swaps words with synonyms
#                          "profits rose"  → "earnings rose"
#
#   RandomWordAug       →  deletes random words
#   (delete)               "profits rose sharply" → "profits sharply"
#
#   RandomWordAug       →  swaps word positions
#   (swap)                 "profits rose sharply" → "rose profits sharply"
#
#   KeyboardAug         →  injects realistic typos
#                          "profits" → "profkts"
#
#   The TEST SET is never touched — we always evaluate on real data.

print("\nLoading augmenters...")
augmenters = [
    naw.SynonymAug(),                    # swap words with synonyms
    naw.RandomWordAug(action="delete"),  # delete random words
    naw.RandomWordAug(action="swap"),    # swap word positions
    nac.KeyboardAug(),                   # inject keyboard typos
]

print("Augmenting training data (this may take a moment)...")
aug_texts, aug_labels = [], []

for text, label in zip(train_texts, train_labels):
    aug      = random.choice(augmenters)  # pick a random technique
    new_text = aug.augment(text)[0]       # apply it
    aug_texts.append(new_text)
    aug_labels.append(label)              # label NEVER changes!

# Combine original + augmented
train_texts  = train_texts  + aug_texts
train_labels = train_labels + aug_labels

# Shuffle so originals and augmented are interleaved
combined = list(zip(train_texts, train_labels))
random.shuffle(combined)
train_texts, train_labels = zip(*combined)
train_texts  = list(train_texts)
train_labels = list(train_labels)

print(f"Training rows after augmentation: {len(train_texts)}  (doubled!)")


# STEP 2: tf.data pipelines
def make_dataset(texts, labels, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((texts, labels))
    if shuffle:
        ds = ds.shuffle(len(texts), seed=SEED)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

train_ds = make_dataset(train_texts, train_labels, shuffle=True)
test_ds  = make_dataset(test_texts,  test_labels,  shuffle=False)


# STEP 3: TextVectorization
vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode="int",
    output_sequence_length=MAX_LEN,
    standardize="lower_and_strip_punctuation"
)
vectorizer.adapt(train_texts)
print(f"\nVocabulary: {len(vectorizer.get_vocabulary())} tokens")

def vectorize(text, label):
    return vectorizer(text), label

train_ds = train_ds.map(vectorize)
test_ds  = test_ds.map(vectorize)


# STEP 4: Build BiSimpleRNN model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(VOCAB_SIZE, 64, input_length=MAX_LEN),
    tf.keras.layers.Bidirectional(tf.keras.layers.SimpleRNN(64)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(3, activation="softmax")
], name="BiSimpleRNN")

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
model.summary()


# STEP 5: Class weights + Train
weights = compute_class_weight(
    class_weight="balanced",
    classes=np.array([0, 1, 2]),
    y=train_labels
)
class_weight = {0: weights[0], 1: weights[1], 2: weights[2]}

print(f"\nClass weights: bearish={weights[0]:.2f}, bullsh={weights[1]:.2f}, neutral={weights[2]:.2f}")
print("\nTraining BiSimpleRNN...")
t0 = time.time()
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    verbose=1,
    class_weight=class_weight
)
train_time = time.time() - t0
print(f"\nDone in {train_time:.0f}s")


# STEP 6: Evaluate
print("\nEvaluating...")
y_true, y_pred = [], []
for text_batch, label_batch in test_ds:
    preds = model.predict(text_batch, verbose=0).argmax(axis=1)
    y_pred.extend(preds.tolist())
    y_true.extend(label_batch.numpy().tolist())

print("\n" + "─" * 52)
print("CLASSIFICATION REPORT  (BiSimpleRNN + nlpaug)")
print("─" * 52)
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))
print(f"Test accuracy : {accuracy_score(y_true, y_pred)*100:.2f}%")
print(f"Best train acc: {max(history.history['accuracy'])*100:.2f}%")
print(f"Training time : {train_time:.0f}s")
print(f"Parameters    : {model.count_params():,}")


# STEP 7: Predict any sentence
def predict(sentence):
    probs = model.predict(vectorizer([sentence]), verbose=0)[0]
    idx   = probs.argmax()
    print(f"  Text  : {sentence[:70]}")
    print(f"  Result: {CLASS_NAMES[idx].upper()}  ({probs[idx]*100:.0f}% confident)")
    print(f"  Scores: bear={probs[0]:.2f}  bull={probs[1]:.2f}  neu={probs[2]:.2f}\n")

print("─" * 52)
print("SAMPLE PREDICTIONS  (BiSimpleRNN + nlpaug)")
print("─" * 52)
predict("Apple stock surges to all-time high on record earnings")
predict("Markets crash as recession fears grip investors worldwide")
predict("Fed holds rates steady as inflation remains uncertain")
predict("Tesla shares drop 15% after missing delivery targets badly")
predict("Goldman Sachs upgrades sector outlook citing strong growth")
