import csv
import random
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight

# ============================================================
# CHOOSE MODEL TYPE HERE
#   1 = Bidirectional SimpleRNN
#   2 = Bidirectional GRU
# ============================================================
MODEL_TYPE = 1   # <-- change this to 1 or 2

# CONFIG
DATA_PATH  = "/data/input/all-data.csv"
BATCH_SIZE = 64
VOCAB_SIZE = 5000
MAX_LEN    = 50
EPOCHS     = 10

LABEL_MAP  = {"negative": 0, "neutral": 1, "positive": 2}


# STEP 1: Read all-data.csv and split 80 / 20
texts, labels = [], []
with open(DATA_PATH, encoding="latin-1") as f:
    for row in csv.reader(f):
        if len(row) >= 2:
            labels.append(LABEL_MAP.get(row[0].strip(), 1))
            texts.append(row[1].strip())

random.seed(42)
indices = list(range(len(texts)))
random.shuffle(indices)

split        = int(len(indices) * 0.8)
train_idx    = indices[:split]
test_idx     = indices[split:]

train_texts  = [texts[i]  for i in train_idx]
train_labels = [labels[i] for i in train_idx]
test_texts   = [texts[i]  for i in test_idx]
test_labels  = [labels[i] for i in test_idx]

N_TRAIN          = len(train_texts)
N_TEST           = len(test_texts)
STEPS_PER_EPOCH  = N_TRAIN // BATCH_SIZE
VALIDATION_STEPS = N_TEST  // BATCH_SIZE + 1

print(f"Total rows   : {len(texts)}")
print(f"Training rows: {N_TRAIN}  |  Test rows: {N_TEST}")


# STEP 2: make_dataset()
def make_dataset(text_list, label_list, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices((text_list, label_list))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=N_TRAIN)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

train_ds = make_dataset(train_texts, train_labels, shuffle=True)
test_ds  = make_dataset(test_texts,  test_labels,  shuffle=False)

print("tf.data pipelines ready")


# STEP 3: TextVectorization
vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode="int",
    output_sequence_length=MAX_LEN
)

vectorizer.adapt(train_texts)
print(f"Vocabulary built: {len(vectorizer.get_vocabulary())} tokens")

def vectorize(text, label):
    return vectorizer(text), label

train_ds = train_ds.map(vectorize)
test_ds  = test_ds.map(vectorize)


# STEP 4: Build the model
#   MODEL_TYPE = 1  ->  Bidirectional SimpleRNN
#   MODEL_TYPE = 2  ->  Bidirectional GRU
#
#   Embedding     ->  word ID becomes a 32-number dense vector
#   Bidirectional ->  reads the sentence left->right AND right->left,
#                      then joins both outputs (32 + 32 = 64 values)
#   Dropout       ->  randomly zeros 30% of values during training
#                      to reduce overfitting
#   Dense         ->  outputs probability for each of the 3 classes

if MODEL_TYPE == 1:
    rnn_layer  = tf.keras.layers.SimpleRNN(32)
    model_name = "BiRNN"
elif MODEL_TYPE == 2:
    rnn_layer  = tf.keras.layers.GRU(32, recurrent_dropout=0.2)
    model_name = "BiGRU"
else:
    raise ValueError("MODEL_TYPE must be 1 (SimpleRNN) or 2 (GRU)")

print(f"\nSelected model: {model_name}  (MODEL_TYPE={MODEL_TYPE})")

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=VOCAB_SIZE, output_dim=32, input_length=MAX_LEN),
    tf.keras.layers.Bidirectional(rnn_layer),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(3, activation="softmax")
], name=model_name)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# STEP 5: Class weighting + Train
#
#   The dataset is imbalanced (e.g. neutral is the majority class).
#   Without weighting, the model ignores rare classes to maximise accuracy.
#   compute_class_weight("balanced") assigns weights inversely proportional
#   to frequency, so a mistake on a rare class costs proportionally more.

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.array([0, 1, 2]),
    y=train_labels
)
class_weight = {0: weights[0], 1: weights[1], 2: weights[2]}

print(f"\nClass weights: negative={weights[0]:.2f}, neutral={weights[1]:.2f}, positive={weights[2]:.2f}")
print("\nTraining...")

history = model.fit(
    train_ds,
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=EPOCHS,
    validation_data=test_ds,
    validation_steps=VALIDATION_STEPS,
    class_weight=class_weight,
    verbose=1
)


# STEP 6: Evaluate — Precision, Recall, F1, Support
print("\nEvaluating...")

y_true, y_pred = [], []
for text_batch, label_batch in test_ds.take(VALIDATION_STEPS):
    preds = model.predict(text_batch, verbose=0).argmax(axis=1)
    y_pred.extend(preds.tolist())
    y_true.extend(label_batch.numpy().tolist())

y_true = y_true[:N_TEST]
y_pred = y_pred[:N_TEST]

print("\n" + "─" * 52)
print(f"CLASSIFICATION REPORT  ({model_name})")
print("─" * 52)
print(classification_report(
    y_true, y_pred,
    target_names=["negative", "neutral", "positive"],
    zero_division=0
))


# STEP 7: Predict any sentence
def predict(sentence):
    vec   = vectorizer([sentence])
    probs = model.predict(vec, verbose=0)[0]
    idx   = probs.argmax()
    label = ["negative", "neutral", "positive"][idx]
    conf  = probs[idx] * 100
    print(f"  Text  : {sentence[:70]}")
    print(f"  Result: {label.upper()}  ({conf:.0f}% confident)\n")

print("─" * 52)
print("SAMPLE PREDICTIONS")
print("─" * 52)
predict("The company reported record profits and strong revenue growth")
predict("Massive layoffs announced as sales decline sharply")
predict("The firm maintained normal operations this quarter")
