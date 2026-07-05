# ─────────────────────────────────────────────────────────────────────
# STEP 0: SETTINGS — change these two lines if needed, nothing else
# ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR = r"C:\alaa\tmu\project\finanace-example\docker\docker\data\output"
# ^ Point this at the TOP-LEVEL output folder (the one containing
#   "checkpoints" and "vocab.txt" directly) -- NOT a worker-0 subfolder.
#   Only the chief writes checkpoints/vocab.txt/etc., so they live one
#   level up from the per-worker log folders.

MODEL_TYPE = 1
# ^ Must match what you used for training:
#   1 = BiSimpleRNN   |   2 = BiGRU

# ─────────────────────────────────────────────────────────────────────
# STEP 1: Imports
# ─────────────────────────────────────────────────────────────────────
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# ^ Must be set BEFORE importing tensorflow, or it has no effect.

import glob
import numpy as np
import tensorflow as tf

print("Step 1: Tools loaded")

# ─────────────────────────────────────────────────────────────────────
# STEP 1b: Hyperparameters — must match training exactly, per MODEL_TYPE
# ─────────────────────────────────────────────────────────────────────
# NOTE: embedding_dim=100 is GloVe's closest available size to the
# tuned 128 -- GloVe only ships in 50/100/200/300 dimensions. Both
# configs below share identical values as given; if that's not what you
# intended for BiSimpleRNN specifically, double-check your sweep results.
MODEL_CONFIGS = {
    1: {
        "name": "BiSimpleRNN",
        "learning_rate": 0.002,
        "vocab_size": 15000,
        "embedding_dim": 100,
        "rnn_units": 128,
        "dropout_rate": 0.20,
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

# Sort by epoch number embedded in the filename (ckpt_01, ckpt_02, ... ckpt_12)
ckpt_index_files.sort()
latest_index_file = ckpt_index_files[-1]
CHECKPOINT_PATH = latest_index_file[:-len(".index")]  # strip ".index" suffix

print(f"Model type      : {MODEL_TYPE} ({model_name})")
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
    """
    Learns a weight for each timestep of a BiRNN's output, then returns a
    single weighted-sum vector per sequence.
    """
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

# The real GloVe vectors aren't needed here -- loading the checkpoint in
# Step 5 overwrites every weight (including the embedding layer) with the
# actual trained values. A zero placeholder of the right shape is enough
# to construct the model, and it means this script needs no GloVe
# download at all.
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
# NOTE: these checkpoints were saved in the older TensorFlow checkpoint
# format (ckpt_09.index + ckpt_09.data-...) via ModelCheckpoint with
# save_weights_only=True. Keras 3's model.load_weights() no longer reads
# that format directly (it only reads .weights.h5/.keras), so we load it
# the way TensorFlow checkpoints are meant to be restored.
checkpoint = tf.train.Checkpoint(model)
checkpoint.restore(CHECKPOINT_PATH).expect_partial()
print("Step 5: Weights loaded! The model is now trained")

# ─────────────────────────────────────────────────────────────────────
# STEP 6: Try it out — predict on some example sentences
# ─────────────────────────────────────────────────────────────────────
print("\nStep 6: Predictions:\n")

# The model outputs 3 numbers (probabilities) for each sentence:
#   [bearish_chance, bullish_chance, neutral_chance]
LABELS = ["BEARISH", "BULLISH", "NEUTRAL"]

# Put any sentences you want to test here:
my_sentences = [
    "Apple stock surges to all-time high on record earnings",
    "Markets crash as recession fears grip investors worldwide",
    "Fed holds rates steady as inflation remains uncertain",
    "Tesla shares drop 15% after missing delivery targets badly",
    "Goldman Sachs upgrades sector outlook citing strong growth",
]

for sentence in my_sentences:
    numbers = vectorizer([sentence])                       # turn text into numbers
    probabilities = model.predict(numbers, verbose=0)[0]    # ask the model
    best_guess = probabilities.argmax()                     # pick the highest score

    print(f"Sentence : {sentence}")
    print(f"Guess    : {LABELS[best_guess]}  ({probabilities[best_guess]*100:.0f}% sure)")
    print("")

print("Done")
