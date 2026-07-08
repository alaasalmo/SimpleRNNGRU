#  BiSimpleRNN / BiGRU — Inference script (Docker)
#  Loads a trained checkpoint + vocab produced by the training script and
#  runs a handful of example predictions.
#
#  Usage:
#    python predict_twitter_sentiment.py --output /data/output --model-type 1
#
#  MODEL_TYPE:
#    1 = BiSimpleRNN   |   2 = BiGRU   (must match what you trained)
#
# ─────────────────────────────────────────────────────────────────────
# STEP 0: Settings — via CLI args / env vars (containerized, no hardcoded
# Windows paths). Falls back to env vars OUTPUT_DIR / MODEL_TYPE if the
# corresponding flag isn't passed, so it can be driven entirely from a
# Kubernetes ConfigMap the same way the training Job is.
# ─────────────────────────────────────────────────────────────────────
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# ^ Must be set BEFORE importing tensorflow, or it has no effect.

import argparse
import glob
import numpy as np
import tensorflow as tf

parser = argparse.ArgumentParser(description="BiSimpleRNN/BiGRU inference (Twitter financial sentiment)")
parser.add_argument("--output", default=os.environ.get("OUTPUT_DIR", "/data/output"),
                     help="Top-level output dir containing 'checkpoints/' and 'vocab.txt' (default: /data/output or $OUTPUT_DIR)")
parser.add_argument("--model-type", type=int, default=int(os.environ.get("MODEL_TYPE", "1")), choices=[1, 2],
                     help="1 = BiSimpleRNN, 2 = BiGRU — must match training (default: 1 or $MODEL_TYPE)")
args = parser.parse_args()

OUTPUT_DIR = args.output
MODEL_TYPE = args.model_type

print("Step 1: Tools loaded")

# ─────────────────────────────────────────────────────────────────────
# STEP 1b: Hyperparameters — must match training exactly, per MODEL_TYPE
# ─────────────────────────────────────────────────────────────────────
MODEL_CONFIGS = {
    1: {
        "name": "BiSimpleRNN",
        "learning_rate": 0.0005,
        "vocab_size": 6000,
        "embedding_dim": 100,
        "rnn_units": 64,
        "dropout_rate": 0.10,
        "max_len": 60,
    },
    2: {
        "name": "BiGRU",
        "learning_rate": 0.002,
        "vocab_size": 15000,
        "embedding_dim": 100,
        "rnn_units": 128,
        "dropout_rate": 0.20,
        "max_len": 60,
    },
}
HP = MODEL_CONFIGS[MODEL_TYPE]
model_name = HP["name"]

# ─────────────────────────────────────────────────────────────────────
# STEP 1c: Find the checkpoint and vocab files automatically
# ─────────────────────────────────────────────────────────────────────
VOCAB_PATH = os.path.join(OUTPUT_DIR, "vocab.txt")

ckpt_index_files = glob.glob(os.path.join(OUTPUT_DIR, "checkpoints", "ckpt_*.index"))
if not ckpt_index_files:
    raise SystemExit(
        f"No checkpoints found in {os.path.join(OUTPUT_DIR, 'checkpoints')} — "
        f"did training finish and save at least one epoch?"
    )

ckpt_index_files.sort()
latest_index_file = ckpt_index_files[-1]
CHECKPOINT_PATH = latest_index_file[:-len(".index")]  # strip ".index" suffix

print(f"Model type      : {MODEL_TYPE} ({model_name})")
print(f"Output dir       : {OUTPUT_DIR}")
print(f"Found checkpoint: {CHECKPOINT_PATH}")
print(f"Found vocab file: {VOCAB_PATH}")

# ─────────────────────────────────────────────────────────────────────
# STEP 2: Rebuild the vectorizer from the saved vocabulary
# ─────────────────────────────────────────────────────────────────────
print("Step 2: Loading saved vocabulary...")

with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    saved_vocab = [line.rstrip("\n") for line in f]

vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=HP["vocab_size"],
    output_mode="int",
    output_sequence_length=HP["max_len"],
    standardize="lower_and_strip_punctuation",
    vocabulary=saved_vocab
)

print(f"Step 2: Done. Vocabulary has {len(vectorizer.get_vocabulary())} words")

# ─────────────────────────────────────────────────────────────────────
# STEP 3: Attention pooling (same function used during training)
# ─────────────────────────────────────────────────────────────────────
def attention_pooling(inputs):
    hidden_dim = inputs.shape[-1]
    scores = tf.keras.layers.Dense(hidden_dim, activation="tanh")(inputs)
    scores = tf.keras.layers.Dense(1)(scores)
    weights = tf.keras.layers.Softmax(axis=1)(scores)
    weighted = tf.keras.layers.Multiply()([inputs, weights])
    pooled = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(weighted)
    return pooled

# ─────────────────────────────────────────────────────────────────────
# STEP 4: Build an empty model with the same shape as the trained one
# ─────────────────────────────────────────────────────────────────────
print(f"Step 4: Building empty {model_name} model shape...")

placeholder_embedding = np.zeros((HP["vocab_size"], HP["embedding_dim"]), dtype="float32")

if MODEL_TYPE == 1:
    rnn_layer = tf.keras.layers.SimpleRNN(HP["rnn_units"], return_sequences=True)
else:
    rnn_layer = tf.keras.layers.GRU(HP["rnn_units"], return_sequences=True)

inputs = tf.keras.Input(shape=(HP["max_len"],), dtype="int32")
x = tf.keras.layers.Embedding(
    HP["vocab_size"], HP["embedding_dim"],
    embeddings_initializer=tf.keras.initializers.Constant(placeholder_embedding),
    trainable=False,
)(inputs)
x = tf.keras.layers.Bidirectional(rnn_layer)(x)
x = attention_pooling(x)
x = tf.keras.layers.Dropout(HP["dropout_rate"])(x)
x = tf.keras.layers.Dense(32, activation="relu")(x)
outputs = tf.keras.layers.Dense(3, activation="softmax")(x)
model = tf.keras.Model(inputs, outputs, name=model_name)

print("Step 4: Empty model built")

# ─────────────────────────────────────────────────────────────────────
# STEP 5: Load the trained weights into the empty model
# ─────────────────────────────────────────────────────────────────────
print(f"Step 5: Loading saved weights from: {CHECKPOINT_PATH}")
checkpoint = tf.train.Checkpoint(model)
checkpoint.restore(CHECKPOINT_PATH).expect_partial()
print("Step 5: Weights loaded! The model is now trained")

# ─────────────────────────────────────────────────────────────────────
# STEP 6: Try it out — predict on some example sentences
# ─────────────────────────────────────────────────────────────────────
print("\nStep 6: Predictions:\n")

LABELS = ["BEARISH", "BULLISH", "NEUTRAL"]

my_sentences = [
    "Apple stock surges to all-time high on record earnings",
    "Markets crash as recession fears grip investors worldwide",
    "Fed holds rates steady as inflation remains uncertain",
    "Tesla shares drop 15% after missing delivery targets badly",
    "Goldman Sachs upgrades sector outlook citing strong growth",
]

for sentence in my_sentences:
    numbers = vectorizer([sentence])
    probabilities = model.predict(numbers, verbose=0)[0]
    best_guess = probabilities.argmax()

    print(f"Sentence : {sentence}")
    print(f"Guess    : {LABELS[best_guess]}  ({probabilities[best_guess]*100:.0f}% sure)")
    print("")

print("Done")
