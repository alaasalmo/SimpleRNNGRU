#  BiSimpleRNN / BiGRU — Augmentation -> GloVe -> Attention -> Dense -> Softmax
#  Financial Sentiment Analysis — Multi-Worker Mirrored Strategy (Docker)
#  Dataset: zeroshot/twitter-financial-news-sentiment (HuggingFace)
#
#  Usage:
#    python birnn_bigru_glove_attention_twitter_windows.py --input /data/input --output /data/output --model-type 1
#
#  MODEL_TYPE:
#    1 = Bidirectional SimpleRNN  (dropout=0.10, epochs=9,  lr=0.0005, vocab=6000,  rnn_units=64)
#    2 = Bidirectional GRU        (dropout=0.20, epochs=12, lr=0.002,  vocab=15000, rnn_units=128)
#
#  Both configs use embedding_dim=100 (GloVe's closest available size to the
#  tuned 64/128 values — GloVe only ships in 50/100/200/300 dimensions).
#  EPOCHS are fixed at each model's already-known best epoch — no early
#  stopping needed here, since that search was already done separately.
#
#  TF_CONFIG is injected via environment variable (env-file per worker).

import os, sys, json, random, time, argparse, zipfile, urllib.request
from datetime import datetime
import numpy as np
import tensorflow as tf
from datasets import load_dataset
from sklearn.metrics import classification_report, accuracy_score, f1_score

import warnings
warnings.filterwarnings("ignore")

# On Windows, stdout/stderr default to the legacy system codepage (e.g. cp1252),
# which can't encode the Unicode box-drawing characters Keras' model.summary()
# and this script's own "─" separators use. Force UTF-8 so nothing crashes on
# print(), regardless of platform.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

import nltk
from nltk.corpus import wordnet

# ── ARGS ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="BiSimpleRNN/BiGRU + GloVe + Attention MultiWorker Training (Twitter)")
parser.add_argument("--input",       required=True, help="Input directory (dataset/GloVe/NLTK cache)")
parser.add_argument("--output",      required=True, help="Output directory (logs, checkpoints, report)")
parser.add_argument("--model-type",  type=int, default=1, choices=[1, 2],
                    help="1 = Bidirectional SimpleRNN, 2 = Bidirectional GRU (default: 1)")
parser.add_argument("--start-delay", type=int, default=10,
                    help="Seconds non-chief workers wait before connecting (default: 10)")
args = parser.parse_args()

INPUT_DIR  = args.input
OUTPUT_DIR = args.output
MODEL_TYPE = args.model_type
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── CONFIG (per model type) ───────────────────────────────────────────────────
SEED        = 42
BATCH_SIZE  = 32
CLASS_NAMES = ["bearish", "bullish", "neutral"]

MODEL_CONFIGS = {
    1: {
        "name": "BiSimpleRNN",
        "epochs": 9,             # fixed - best epoch found earlier
        "learning_rate": 0.0005,
        "vocab_size": 6000,
        "embedding_dim": 100,     # GloVe's closest size to the tuned 64
        "rnn_units": 64,
        "dropout_rate": 0.10,
        "max_len": 60,
    },
    2: {
        "name": "BiGRU",
        "epochs": 12,            # fixed - best epoch found earlier
        "learning_rate": 0.002,
        "vocab_size": 15000,
        "embedding_dim": 100,     # GloVe's closest size to the tuned 128
        "rnn_units": 128,
        "dropout_rate": 0.20,
        "max_len": 60,
    },
}

HP = MODEL_CONFIGS[MODEL_TYPE]
EPOCHS = HP["epochs"]
model_name = HP["name"]

AUGMENT_FRACTION = 0.30
AUGMENT_WORD_PROB = 0.20

STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "if", "is", "are", "was", "were",
    "be", "been", "being", "to", "of", "in", "on", "for", "with", "as",
    "at", "by", "from", "this", "that", "these", "those", "it", "its",
    "i", "you", "he", "she", "we", "they", "them", "his", "her", "their",
    "our", "your", "my", "me", "him", "us", "not", "no", "do", "does",
    "did", "will", "would", "shall", "should", "can", "could", "may",
    "might", "must", "have", "has", "had", "so", "than", "then", "there",
    "here", "up", "down", "out", "over", "under", "again", "further",
    "once", "s", "t", "re", "ve", "ll", "d", "m",
}

BAKED_GLOVE_FILE = os.path.join(os.environ.get("GLOVE_DIR", "/usr/local/glove"), "glove.6B.100d.txt")
GLOVE_DIR = os.path.join(INPUT_DIR, "glove")   # fallback: shared with other workers via mounted volume
GLOVE_FILE = os.path.join(GLOVE_DIR, "glove.6B.100d.txt")
GLOVE_ZIP_URL = "https://nlp.stanford.edu/data/glove.6B.zip"

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ── READ TF_CONFIG EARLY ──────────────────────────────────────────────────────
tf_config  = json.loads(os.environ.get("TF_CONFIG", "{}"))
task_type  = tf_config.get("task", {}).get("type", "worker")
task_index = tf_config.get("task", {}).get("index", 0)
is_chief   = (task_type == "worker" and task_index == 0)

# ── LOGGING ───────────────────────────────────────────────────────────────────
# Each worker gets its own subfolder under OUTPUT_DIR (worker-0, worker-1, ...).
# Without this, every worker would write "train.log" to the exact same file
# if OUTPUT_DIR points at a shared host folder mounted into multiple
# containers -- they'd overwrite/interleave each other's logs.
WORKER_OUTPUT_DIR = os.path.join(OUTPUT_DIR, f"worker-{task_index}")
os.makedirs(WORKER_OUTPUT_DIR, exist_ok=True)

log_path = os.path.join(WORKER_OUTPUT_DIR, "train.log")
log_file = open(log_path, "w", buffering=1, encoding="utf-8")

def log(msg=""):
    print(msg, flush=True)
    log_file.write(msg + "\n")

run_start_time = time.time()
run_start_dt = datetime.now()

log(f"{'='*52}")
log(f"{model_name}  |  worker-{task_index}  ({'chief' if is_chief else 'worker'})")
log(f"{'='*52}")
log(f"[Run]      start      = {run_start_dt:%Y-%m-%d %H:%M:%S}")
log(f"[Config]   input      = {INPUT_DIR}")
log(f"[Config]   output     = {OUTPUT_DIR}")
log(f"[Config]   model_type = {MODEL_TYPE} ({model_name})")
log(f"[Config]   HP         = {HP}")

# ── STARTUP DELAY (non-chief only) ────────────────────────────────────────────
if not is_chief:
    delay = args.start_delay if args.start_delay > 0 else 10
    log(f"[Startup]  Waiting {delay}s for chief to be ready…")
    time.sleep(delay)
    log(f"[Startup]  Done waiting ✓")

# ── STRATEGY ──────────────────────────────────────────────────────────────────
# Must be created before ANY other TensorFlow op touches the runtime (this is
# a hard TensorFlow requirement — "Collective ops must be configured at
# program startup"). To avoid the coordination-service heartbeat timing out
# during the slow GloVe download/dataset steps below, GloVe and NLTK data are
# pre-baked into the Docker image at build time instead (see Dockerfile), so
# nothing here blocks for more than a few seconds.
log(f"\n[Strategy] Initialising MultiWorkerMirroredStrategy…")
strategy = tf.distribute.MultiWorkerMirroredStrategy()
log(f"[Strategy] replicas = {strategy.num_replicas_in_sync}")
log(f"[Worker]   type={task_type}  index={task_index}  is_chief={is_chief}")

# ── STEP 1: Load dataset ──────────────────────────────────────────────────────
os.environ["HF_HOME"]           = INPUT_DIR
os.environ["HF_DATASETS_CACHE"] = os.path.join(INPUT_DIR, "datasets")

log("\n[Step 1] Loading dataset…")
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
log(f"Train: {len(train_texts)} | Test: {len(test_texts)}")

# ── STEP 2: Augmentation (WordNet synonym replacement, on raw text) ──────────
log("\n[Step 2] Augmenting training text (WordNet synonym replacement)…")

def ensure_nltk_data():
    nltk_local_dir = os.path.join(INPUT_DIR, "nltk_data")
    if nltk_local_dir not in nltk.data.path:
        nltk.data.path.append(nltk_local_dir)
    for pkg in ["wordnet", "omw-1.4"]:
        try:
            nltk.data.find(f"corpora/{pkg}")
        except LookupError:
            log(f"Downloading NLTK '{pkg}' corpus…")
            nltk.download(pkg, quiet=True, download_dir=nltk_local_dir)

def get_synonym(word):
    if word.lower() in STOP_WORDS or not word.isalpha():
        return None
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            candidate = lemma.name().replace("_", " ")
            if candidate.lower() != word.lower():
                synonyms.add(candidate)
    return random.choice(list(synonyms)) if synonyms else None

def augment_text(text, word_prob=AUGMENT_WORD_PROB):
    words = text.split()
    new_words = []
    for word in words:
        if random.random() < word_prob:
            synonym = get_synonym(word)
            new_words.append(synonym if synonym else word)
        else:
            new_words.append(word)
    return " ".join(new_words)

def augment_dataset(texts, labels, fraction=AUGMENT_FRACTION):
    ensure_nltk_data()
    n_to_augment = int(len(texts) * fraction)
    indices = random.sample(range(len(texts)), n_to_augment)
    aug_texts = [augment_text(texts[i]) for i in indices]
    aug_labels = [labels[i] for i in indices]
    return texts + aug_texts, labels + aug_labels

train_texts, train_labels = augment_dataset(train_texts, train_labels)
log(f"Train (post-augmentation): {len(train_texts)}")

# ── STEP 3: TextVectorization ─────────────────────────────────────────────────
log("\n[Step 3] Adapting vectorizer…")
vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=HP["vocab_size"],
    output_mode="int",
    output_sequence_length=HP["max_len"],
    standardize="lower_and_strip_punctuation"
)
vectorizer.adapt(train_texts)
log(f"Vocabulary: {len(vectorizer.get_vocabulary())} tokens")

# ── STEP 4: Pretrained GloVe embeddings ───────────────────────────────────────
log("\n[Step 4] Loading pretrained GloVe embeddings…")

def resolve_glove_file():
    """
    Prefers the copy baked into the Docker image at build time (fast, no
    network needed). Falls back to INPUT_DIR/glove (e.g. a pre-populated
    shared volume) and only downloads as a last resort.
    """
    if os.path.exists(BAKED_GLOVE_FILE):
        log(f"Using GloVe vectors baked into the image: {BAKED_GLOVE_FILE}")
        return BAKED_GLOVE_FILE

    if not os.path.exists(GLOVE_FILE):
        os.makedirs(GLOVE_DIR, exist_ok=True)
        zip_path = os.path.join(GLOVE_DIR, "glove.6B.zip")
        log("GloVe file not found in image or input dir — downloading "
            "glove.6B.zip (~800MB). This may be slow enough to trip a "
            "multi-worker heartbeat timeout; consider baking GloVe into "
            "the image instead (see Dockerfile).")
        urllib.request.urlretrieve(GLOVE_ZIP_URL, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extract("glove.6B.100d.txt", GLOVE_DIR)
        os.remove(zip_path)
        log(f"GloVe vectors ready at {GLOVE_FILE}")

    return GLOVE_FILE

def load_glove_vectors(glove_path):
    embeddings = {}
    with open(glove_path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            embeddings[parts[0]] = np.asarray(parts[1:], dtype="float32")
    return embeddings

def build_embedding_matrix(vectorizer, glove_vectors, embedding_dim):
    vocab = vectorizer.get_vocabulary()
    matrix = np.zeros((len(vocab), embedding_dim), dtype="float32")
    hits = 0
    for i, word in enumerate(vocab):
        vector = glove_vectors.get(word)
        if vector is not None:
            matrix[i] = vector
            hits += 1
    log(f"GloVe coverage: {hits}/{len(vocab)} ({hits/len(vocab)*100:.1f}%)")
    return matrix

glove_path = resolve_glove_file()
glove_vectors = load_glove_vectors(glove_path)
embedding_matrix = build_embedding_matrix(vectorizer, glove_vectors, HP["embedding_dim"])

# ── STEP 5: tf.data pipelines ─────────────────────────────────────────────────
global_batch = BATCH_SIZE * strategy.num_replicas_in_sync
log(f"\n[Step 5] Global batch: {global_batch}  ({BATCH_SIZE} x {strategy.num_replicas_in_sync} replicas)")

def make_dataset(texts, labels, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((texts, labels))
    if shuffle:
        ds = ds.shuffle(len(texts), seed=SEED)
    ds = ds.batch(global_batch).prefetch(tf.data.AUTOTUNE)
    return ds.map(lambda t, l: (vectorizer(t), l))

train_ds = make_dataset(train_texts, train_labels, shuffle=True)
test_ds  = make_dataset(test_texts,  test_labels,  shuffle=False)

# ── ATTENTION POOLING (additive/Bahdanau-style, as a function of standard Keras layers) ─
def attention_pooling(inputs):
    """
    Learns a weight for each timestep of a BiRNN's output, then returns a
    single weighted-sum vector per sequence. Built from plain Keras layers
    (Dense, Softmax, Multiply) rather than a custom Layer subclass.
    """
    hidden_dim = inputs.shape[-1]
    scores = tf.keras.layers.Dense(hidden_dim, activation="tanh")(inputs)  # [batch, time, hidden]
    scores = tf.keras.layers.Dense(1)(scores)                             # [batch, time, 1]
    weights = tf.keras.layers.Softmax(axis=1)(scores)                     # [batch, time, 1]
    weighted = tf.keras.layers.Multiply()([inputs, weights])              # [batch, time, hidden]
    pooled = tf.keras.layers.Lambda(
        lambda x: tf.reduce_sum(x, axis=1)
    )(weighted)                                                            # [batch, hidden]
    return pooled

# ── STEP 6: Build model inside strategy scope (switches on MODEL_TYPE) ───────
log(f"\n[Step 6] Building {model_name} model…")

def build_model():
    if MODEL_TYPE == 1:
        rnn_layer = tf.keras.layers.SimpleRNN(HP["rnn_units"], return_sequences=True)
    else:
        rnn_layer = tf.keras.layers.GRU(HP["rnn_units"], return_sequences=True)

    inputs = tf.keras.Input(shape=(HP["max_len"],), dtype="int32")
    x = tf.keras.layers.Embedding(
        HP["vocab_size"], HP["embedding_dim"],
        embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
        trainable=False,
    )(inputs)
    x = tf.keras.layers.Bidirectional(rnn_layer)(x)
    x = attention_pooling(x)
    x = tf.keras.layers.Dropout(HP["dropout_rate"])(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    outputs = tf.keras.layers.Dense(3, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name=model_name)

with strategy.scope():
    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(HP["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

summary_path = os.path.join(OUTPUT_DIR, "model_summary.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    model.summary(print_fn=lambda x: f.write(x + "\n"))
log(f"Model summary → {summary_path}")

if is_chief:
    vocab_path = os.path.join(OUTPUT_DIR, "vocab.txt")
    with open(vocab_path, "w", encoding="utf-8") as f:
        for word in vectorizer.get_vocabulary():
            f.write(word + "\n")
    log(f"Vocabulary saved → {vocab_path}")

# ── STEP 7: Train (fixed EPOCHS per model, static value — no early stopping) ──

ckpt_dir = os.path.join(OUTPUT_DIR, "checkpoints")
tb_dir   = os.path.join(OUTPUT_DIR, "tensorboard")
os.makedirs(ckpt_dir, exist_ok=True)

callbacks = [tf.keras.callbacks.TensorBoard(log_dir=tb_dir)]
if is_chief:
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(ckpt_dir, "ckpt_{epoch:02d}.weights.h5"),
        save_weights_only=True, verbose=1
    ))

log(f"\n[Step 7] Training {model_name} for {EPOCHS} epochs (fixed)…")
t0 = time.time()
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    verbose=1 if is_chief else 0,
    callbacks=callbacks
)
train_time = time.time() - t0
log(f"\nTraining done in {train_time:.0f}s")

# ── Non-chief exits HERE — before any predict() calls ─────────────────────────
# Staying alive during model.predict() would trigger CollectiveReduceV2 ops
# across all workers, causing UnavailableError when other workers disconnect.
if not is_chief:
    run_end_dt = datetime.now()
    run_elapsed = time.time() - run_start_time
    log(f"\n[Worker {task_index}] Training complete — exiting cleanly.")
    log(f"[Run]      end        = {run_end_dt:%Y-%m-%d %H:%M:%S}")
    log(f"[Run]      elapsed    = {run_elapsed:.0f}s ({run_elapsed/60:.1f} min)")
    log_file.close()
    sys.exit(0)

# ── STEP 8: Evaluate — chief only, fresh single-worker inference model ───────
log(f"\n[Step 8] Building single-worker inference model…")
inference_model = build_model()
inference_model.compile(
    optimizer=tf.keras.optimizers.Adam(HP["learning_rate"]),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
inference_model.set_weights(model.get_weights())
log("Weights transferred ✓")

test_ds_infer = (
    tf.data.Dataset.from_tensor_slices((test_texts, test_labels))
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
    .map(lambda t, l: (vectorizer(t), l))
)

log("Evaluating…")
y_true, y_pred = [], []
for text_batch, label_batch in test_ds_infer:
    preds = inference_model.predict(text_batch, verbose=0).argmax(axis=1)
    y_pred.extend(preds.tolist())
    y_true.extend(label_batch.numpy().tolist())

report       = classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0)
acc          = accuracy_score(y_true, y_pred) * 100
macro_f1     = f1_score(y_true, y_pred, average="macro")
combined     = (accuracy_score(y_true, y_pred) + macro_f1) / 2
per_class_f1 = f1_score(y_true, y_pred, average=None)

report_path = os.path.join(OUTPUT_DIR, "classification_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(f"CLASSIFICATION REPORT  (Augmentation -> GloVe -> {model_name} -> Attention — MultiWorker)\n")
    f.write("─" * 52 + "\n")
    f.write(f"Run start       : {run_start_dt:%Y-%m-%d %H:%M:%S}\n")
    f.write(report + "\n")
    f.write(f"Test accuracy   : {acc:.2f}%\n")
    f.write(f"Test macro-F1   : {macro_f1:.4f}\n")
    f.write(f"Test combined   : {combined:.4f}\n")
    f.write(f"Bearish F1: {per_class_f1[0]:.3f} | Bullish F1: {per_class_f1[1]:.3f} | "
            f"Neutral F1: {per_class_f1[2]:.3f}\n")
    f.write(f"Trained epochs  : {EPOCHS} (fixed — best epoch from earlier run)\n")
    f.write(f"Training time   : {train_time:.0f}s\n")
    f.write(f"Parameters      : {model.count_params():,}\n")
    f.write(f"Replicas used   : {strategy.num_replicas_in_sync}\n")
    f.write(f"Model type      : {MODEL_TYPE} ({model_name})\n")

log("\n" + "─" * 52)
log(f"CLASSIFICATION REPORT  (Augmentation -> GloVe -> {model_name} -> Attention — MultiWorker)")
log("─" * 52)
log(report)
log(f"Test accuracy   : {acc:.2f}%")
log(f"Test macro-F1   : {macro_f1:.4f}")
log(f"Test combined   : {combined:.4f}")
log(f"Bearish F1: {per_class_f1[0]:.3f} | Bullish F1: {per_class_f1[1]:.3f} | "
    f"Neutral F1: {per_class_f1[2]:.3f}")
log(f"Trained epochs  : {EPOCHS} (fixed)")
log(f"Training time   : {train_time:.0f}s")
log(f"Parameters      : {model.count_params():,}")
log(f"Replicas used   : {strategy.num_replicas_in_sync}")
log(f"Report saved    → {report_path}")

# ── STEP 9: Sample predictions ────────────────────────────────────────────────
sentences = [
    "Apple stock surges to all-time high on record earnings",
    "Markets crash as recession fears grip investors worldwide",
    "Fed holds rates steady as inflation remains uncertain",
    "Tesla shares drop 15% after missing delivery targets badly",
    "Goldman Sachs upgrades sector outlook citing strong growth",
]

pred_path = os.path.join(OUTPUT_DIR, "sample_predictions.txt")
log("\n" + "─" * 52)
log(f"SAMPLE PREDICTIONS  ({model_name})")
log("─" * 52)

with open(pred_path, "w", encoding="utf-8") as pf:
    pf.write(f"SAMPLE PREDICTIONS  (Augmentation -> GloVe -> {model_name} -> Attention)\n" + "─" * 52 + "\n")
    for sentence in sentences:
        probs = inference_model.predict(vectorizer([sentence]), verbose=0)[0]
        idx   = probs.argmax()
        line  = (f"  Text  : {sentence}\n"
                 f"  Result: {CLASS_NAMES[idx].upper()}  ({probs[idx]*100:.0f}% confident)\n"
                 f"  Scores: bear={probs[0]:.2f}  bull={probs[1]:.2f}  neu={probs[2]:.2f}\n")
        log(line)
        pf.write(line + "\n")

log(f"Predictions saved → {pred_path}")
run_end_dt = datetime.now()
run_elapsed = time.time() - run_start_time
log(f"\n[Run]      end        = {run_end_dt:%Y-%m-%d %H:%M:%S}")
log(f"[Run]      elapsed    = {run_elapsed:.0f}s ({run_elapsed/60:.1f} min)")
log(f"\n[Worker 0] All done. Outputs in: {OUTPUT_DIR}")
log_file.close()
