#  GRU — Financial Sentiment Analysis
#  Dataset: zeroshot/twitter-financial-news-sentiment

import random, time
import numpy as np
import tensorflow as tf
from datasets import load_dataset
from sklearn.metrics import classification_report, accuracy_score
from collections import Counter

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

train_counts = Counter(train_labels)
test_counts = Counter(test_labels)

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
print(f"Vocabulary: {len(vectorizer.get_vocabulary())} tokens")

def vectorize(text, label):
    return vectorizer(text), label

train_ds = train_ds.map(vectorize)
test_ds  = test_ds.map(vectorize)


# STEP 4: Build GRU model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(VOCAB_SIZE, 64),
    tf.keras.layers.GRU(64),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(3, activation="softmax")
], name="GRU")

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
model.summary()


# STEP 5: Train
print("\nTraining GRU...")
t0 = time.time()
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    verbose=1
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
print("CLASSIFICATION REPORT  (GRU)")
print("─" * 52)
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))
print(f"Test accuracy : {accuracy_score(y_true, y_pred)*100:.2f}%")
print(f"Best train acc: {max(history.history['accuracy'])*100:.2f}%")
print(f"Training time : {train_time:.0f}s")
print(f"Parameters    : {model.count_params():,}")

print("\nTraining Set Distribution")
print("Bearish:", train_counts[0])
print("Bullish :", train_counts[1])
print("Neutral:", train_counts[2])
print("\nTest Set Distribution")
print("Bearish:", test_counts[0])
print("Bullish :", test_counts[1])
print("Neutral:", test_counts[2])

# STEP 7: Predict any sentence
def predict(sentence):
    probs = model.predict(vectorizer([sentence]), verbose=0)[0]
    idx   = probs.argmax()
    print(f"  Text  : {sentence[:70]}")
    print(f"  Result: {CLASS_NAMES[idx].upper()}  ({probs[idx]*100:.0f}% confident)")
    print(f"  Scores: bear={probs[0]:.2f}  bull={probs[1]:.2f}  neu={probs[2]:.2f}\n")

print("─" * 52)
print("SAMPLE PREDICTIONS  (GRU)")
print("─" * 52)
predict("Apple stock surges to all-time high on record earnings")
predict("Markets crash as recession fears grip investors worldwide")
predict("Fed holds rates steady as inflation remains uncertain")
predict("Tesla shares drop 15% after missing delivery targets badly")
predict("Goldman Sachs upgrades sector outlook citing strong growth")