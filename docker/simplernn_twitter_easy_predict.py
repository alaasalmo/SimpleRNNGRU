# ─────────────────────────────────────────────────────────────────────
# STEP 0: SETTINGS — change these two lines if needed, nothing else
# ─────────────────────────────────────────────────────────────────────
##WORKER_DIR = "docker/data/output/worker-0"
WORKER_DIR = r"C:\alaa\github\SimpleRNNGRU\docker\data\output\worker-0"
# ^ Just point this at the worker-0 output folder. The script will
#   automatically find the latest checkpoint and vocab.txt inside it.

MODEL_TYPE = 1
# ^ Must match what you used for training:
#   1 = SimpleRNN model   |   2 = GRU model

# ─────────────────────────────────────────────────────────────────────
# STEP 1: Imports (just loading the tools we need)
# ─────────────────────────────────────────────────────────────────────
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# ^ Must be set BEFORE importing tensorflow, or it has no effect.

import glob
import tensorflow as tf

print("Step 1: Tools loaded")

# ─────────────────────────────────────────────────────────────────────
# STEP 1b: Find the checkpoint and vocab files automatically
# ─────────────────────────────────────────────────────────────────────
VOCAB_PATH = os.path.join(WORKER_DIR, "vocab.txt")

ckpt_index_files = glob.glob(os.path.join(WORKER_DIR, "checkpoints", "ckpt_*.index"))
if not ckpt_index_files:
    raise SystemExit(
        f"No checkpoints found in {os.path.join(WORKER_DIR, 'checkpoints')} — "
        f"did training finish and save at least one epoch?"
    )

# Sort by epoch number embedded in the filename (ckpt_01, ckpt_02, ... ckpt_10)
ckpt_index_files.sort()
latest_index_file = ckpt_index_files[-1]
CHECKPOINT_PATH = latest_index_file[:-len(".index")]  # strip ".index" suffix

print(f"Found checkpoint: {CHECKPOINT_PATH}")
print(f"Found vocab file: {VOCAB_PATH}")

# ─────────────────────────────────────────────────────────────────────
# STEP 2: Rebuild the vectorizer (turns text into numbers the model understands)
# ─────────────────────────────────────────────────────────────────────
# We read the small vocabulary file saved during training instead of
# re-downloading and re-processing the whole dataset. Same word-to-number
# mapping, instant load, no internet needed.

print("Step 2: Loading saved vocabulary...")

with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    saved_vocab = [line.rstrip("\n") for line in f]

vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=6000,
    output_mode="int",
    output_sequence_length=60,
    standardize="lower_and_strip_punctuation",
    vocabulary=saved_vocab
)

print(f"Step 2: Done. Vocabulary has {len(vectorizer.get_vocabulary())} words")

# ─────────────────────────────────────────────────────────────────────
# STEP 3: Build an empty model with the same shape as the trained one
# ─────────────────────────────────────────────────────────────────────
print("Step 3: Building empty model shape...")

if MODEL_TYPE == 1:
    rnn_layer = tf.keras.layers.SimpleRNN(64)
else:
    rnn_layer = tf.keras.layers.GRU(64)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(6000, 64, input_length=60),
    tf.keras.layers.Bidirectional(rnn_layer),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(3, activation="softmax")
])

print("Step 3: Empty model built")

# ─────────────────────────────────────────────────────────────────────
# STEP 4: Load the trained weights into the empty model
# ─────────────────────────────────────────────────────────────────────
print(f"Step 4: Loading saved weights from: {CHECKPOINT_PATH}")
# NOTE: these checkpoints were saved in the older TensorFlow checkpoint
# format (ckpt_10.index + ckpt_10.data-...) via ModelCheckpoint with
# save_weights_only=True. Keras 3's model.load_weights() no longer reads
# that format directly (it only reads .weights.h5/.keras), so we load it
# the way TensorFlow checkpoints are meant to be restored.
checkpoint = tf.train.Checkpoint(model)
checkpoint.restore(CHECKPOINT_PATH).expect_partial()
print("Step 4: Weights loaded! The model is now trained")

# ─────────────────────────────────────────────────────────────────────
# STEP 5: Try it out — predict on some example sentences
# ─────────────────────────────────────────────────────────────────────
print("\nStep 5: Predictions:\n")

# The model outputs 3 numbers (probabilities) for each sentence:
#   [bearish_chance, bullish_chance, neutral_chance]
LABELS = ["BEARISH", "BULLISH", "NEUTRAL"]

# Put any sentences you want to test here:
my_sentences = [
    "Apple stock surges to all-time high on record earnings",
    "Markets crash as recession fears grip investors worldwide",
    "Fed holds rates steady as inflation remains uncertain",
]

for sentence in my_sentences:
    numbers = vectorizer([sentence])              # turn text into numbers
    probabilities = model.predict(numbers, verbose=0)[0]  # ask the model
    best_guess = probabilities.argmax()            # pick the highest score

    print(f"Sentence : {sentence}")
    print(f"Guess    : {LABELS[best_guess]}  ({probabilities[best_guess]*100:.0f}% sure)")
    print("")

print("Done")
