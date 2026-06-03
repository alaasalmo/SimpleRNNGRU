import csv
import random
import tensorflow as tf
from sklearn.metrics import classification_report

# CONFIG
DATA_PATH  = "all-data.csv"
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


#STEP 2: make_dataset()
# Builds a tf.data pipeline directly from Python lists —
# no CSV files needed at this stage.

def make_dataset(text_list, label_list, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices((text_list, label_list))

    def prepare(features, label):
        text = features["text"]  # raw sentence strings

        # Map string labels → integers: negative=0  neutral=1  positive=2
        label = tf.map_fn(
            lambda x: tf.case(
                [
                    (tf.equal(x, "negative"), lambda: tf.constant(0)),
                    (tf.equal(x, "neutral"),  lambda: tf.constant(1)),
                ],
                default=lambda: tf.constant(2)
            ),
            label,
            fn_output_signature=tf.int32
        )
        return text, label

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
# Learns vocabulary from training text only,
# then converts every sentence to a padded integer sequence.

vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode="int",
    output_sequence_length=MAX_LEN
)

vectorizer.adapt(train_texts)
print(f"Vocabulary built: {len(vectorizer.get_vocabulary())} tokens")

# Apply vectorizer inside the pipeline
def vectorize(text, label):
    return vectorizer(text), label

train_ds = train_ds.map(vectorize)
test_ds  = test_ds.map(vectorize)


# STEP 4: Build the SimpleRNN model
#
#   Embedding  →  word ID → 32-number dense vector
#   SimpleRNN  →  reads the sentence word by word, keeps memory
#   Dense      →  outputs probability for each of the 3 classes

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=VOCAB_SIZE, output_dim=32),
    tf.keras.layers.SimpleRNN(32),
    tf.keras.layers.Dense(3, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# STEP 5: Train

print("\nTraining...")
history = model.fit(
    train_ds,
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=EPOCHS,
    validation_data=test_ds,
    validation_steps=VALIDATION_STEPS,
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
print("CLASSIFICATION REPORT GRU")
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
