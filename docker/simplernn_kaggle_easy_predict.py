# ─────────────────────────────────────────────────────────────────────
# STEP 0: SETTINGS — change these two lines if needed, nothing else
# ─────────────────────────────────────────────────────────────────────
WORKER_DIR = r"C:\alaa\github\SimpleRNNGRU\docker\data\output\worker-0"
# ^ Point this at the worker-0 output folder for the Kaggle training run.
#   The script automatically finds the latest checkpoint and vocab.txt inside it.

MODEL_TYPE = 1
# ^ Must match what you used for training:
#   1 = SimpleRNN model   |   2 = GRU model
#   (this example is set up for SimpleRNN, i.e. --model-type 1)

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
ckpt_dir   = os.path.join(WORKER_DIR, "checkpoints")

ckpt_index_files = glob.glob(os.path.join(ckpt_dir, "ckpt_*.index"))
if not ckpt_index_files:
    abs_worker_dir = os.path.abspath(WORKER_DIR)
    abs_ckpt_dir   = os.path.abspath(ckpt_dir)
    print(f"\n[ERROR] No checkpoints found.")
    print(f"        WORKER_DIR (resolved) = {abs_worker_dir}")
    print(f"        Looked in             = {abs_ckpt_dir}")
    if os.path.isdir(abs_worker_dir):
        print(f"        Contents of {abs_worker_dir}:")
        for name in os.listdir(abs_worker_dir):
            print(f"          - {name}")
    else:
        print(f"        That folder does not exist. Check WORKER_DIR.")
    raise SystemExit("No ckpt_*.index files found — did Kaggle training finish?")

ckpt_index_files.sort()
latest_index_file = ckpt_index_files[-1]
CHECKPOINT_PATH = latest_index_file[:-len(".index")]  # strip ".index" suffix

print(f"Found checkpoint: {CHECKPOINT_PATH}")
print(f"Found vocab file: {VOCAB_PATH}")

# ─────────────────────────────────────────────────────────────────────
# STEP 2: Rebuild the vectorizer (turns text into numbers the model understands)
# ─────────────────────────────────────────────────────────────────────
# We read the small vocabulary file saved during training instead of
# re-reading the whole CSV. Same word-to-number mapping, instant load.

print("Step 2: Loading saved vocabulary...")

with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    saved_vocab = [line.rstrip("\n") for line in f]

vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=5000,            # VOCAB_SIZE — must match Kaggle training script
    output_mode="int",
    output_sequence_length=50,  # MAX_LEN — must match Kaggle training script
    standardize="lower_and_strip_punctuation",
    vocabulary=saved_vocab
)

print(f"Step 2: Done. Vocabulary has {len(vectorizer.get_vocabulary())} words")

# ─────────────────────────────────────────────────────────────────────
# STEP 3: Build an empty model with the same shape as the trained one
# ─────────────────────────────────────────────────────────────────────
# NOTE: this architecture is smaller than the Twitter version — no extra
# Dense(32) layer, since that's what the Kaggle training script uses.
print("Step 3: Building empty model shape...")

if MODEL_TYPE == 1:
    rnn_layer = tf.keras.layers.SimpleRNN(32)
else:
    rnn_layer = tf.keras.layers.GRU(32)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(5000, 32, input_length=50),
    tf.keras.layers.Bidirectional(rnn_layer),
    tf.keras.layers.Dropout(0.3),
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
# that format directly, so we load it via tf.train.Checkpoint instead.
checkpoint = tf.train.Checkpoint(model)
checkpoint.restore(CHECKPOINT_PATH).expect_partial()
print("Step 4: Weights loaded! The model is now trained")

# ─────────────────────────────────────────────────────────────────────
# STEP 5: Try it out — predict on some example sentences
# ─────────────────────────────────────────────────────────────────────
print("\nStep 5: Predictions:\n")

# The model outputs 3 numbers (probabilities) for each sentence:
#   [negative_chance, neutral_chance, positive_chance]
LABELS = ["NEGATIVE", "NEUTRAL", "POSITIVE"]

# Put any sentences you want to test here:
my_sentences = [
    "The company reported record profits and strong revenue growth",
    "Massive layoffs announced as sales decline sharply",
    "The firm maintained normal operations this quarter",
    "Shares tumble after the firm missed earnings expectations",
]

for sentence in my_sentences:
    numbers = vectorizer([sentence])              # turn text into numbers
    probabilities = model.predict(numbers, verbose=0)[0]  # ask the model
    best_guess = probabilities.argmax()            # pick the highest score

    print(f"Sentence : {sentence}")
    print(f"Guess    : {LABELS[best_guess]}  ({probabilities[best_guess]*100:.0f}% sure)")
    print("All scores:")
    for label, prob in zip(LABELS, probabilities):
        marker = " <-- picked" if label == LABELS[best_guess] else ""
        print(f"  {label:8s}: {prob*100:5.1f}%{marker}")
    print("")

print("Done")
