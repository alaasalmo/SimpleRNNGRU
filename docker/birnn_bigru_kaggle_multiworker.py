#  BiSimpleRNN / BiGRU — Financial Sentiment Analysis
#  Dataset: Kaggle "all-data.csv" (negative / neutral / positive)
#  Multi-Worker Mirrored Strategy
#
#  Usage:
#    python bisimplernn_bigru_kaggle_multiworker.py --input /data/input --output /data/output --model-type 1
#
#  MODEL_TYPE:
#    1 = Bidirectional SimpleRNN
#    2 = Bidirectional GRU
#
#  TF_CONFIG is injected via environment variable (env-file per worker).
#
#  --input must point to a directory CONTAINING all-data.csv
#  (e.g. /data/input/all-data.csv)

import os, sys, csv, json, random, time, argparse, traceback
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

# ── ARGS ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="BiSimpleRNN/BiGRU MultiWorker Training (Kaggle CSV)")
parser.add_argument("--input",       required=True, help="Input directory containing all-data.csv")
parser.add_argument("--output",      required=True, help="Output directory (logs, checkpoints, report)")
parser.add_argument("--model-type",  type=int, default=1, choices=[1, 2],
                    help="1 = Bidirectional SimpleRNN, 2 = Bidirectional GRU (default: 1)")
parser.add_argument("--start-delay", type=int, default=10,
                    help="Seconds non-chief workers wait before connecting (default: 10)")
args = parser.parse_args()

INPUT_DIR  = os.path.abspath(args.input)
OUTPUT_DIR = os.path.abspath(args.output)
MODEL_TYPE = args.model_type
DATA_PATH  = os.path.join(INPUT_DIR, "all-data.csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── CONFIG ────────────────────────────────────────────────────────────────────
SEED        = 42
BATCH_SIZE  = 64
VOCAB_SIZE  = 5000
MAX_LEN     = 50
EPOCHS      = 10
CLASS_NAMES = ["negative", "neutral", "positive"]
LABEL_MAP   = {"negative": 0, "neutral": 1, "positive": 2}

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ── READ TF_CONFIG EARLY ──────────────────────────────────────────────────────
tf_config  = json.loads(os.environ.get("TF_CONFIG", "{}"))
task_type  = tf_config.get("task", {}).get("type", "worker")
task_index = tf_config.get("task", {}).get("index", 0)
is_chief   = (task_type == "worker" and task_index == 0)

model_name = "BiSimpleRNN" if MODEL_TYPE == 1 else "BiGRU"

# ── LOGGING ───────────────────────────────────────────────────────────────────
log_path = os.path.join(OUTPUT_DIR, "train.log")
log_file = open(log_path, "w", buffering=1)

def log(msg=""):
    print(msg, flush=True)
    log_file.write(msg + "\n")

log(f"{'='*52}")
log(f"{model_name}  |  worker-{task_index}  ({'chief' if is_chief else 'worker'})")
log(f"{'='*52}")
log(f"[Config]   input      = {INPUT_DIR}")
log(f"[Config]   output     = {OUTPUT_DIR}")
log(f"[Config]   data_path  = {DATA_PATH}")
log(f"[Config]   model_type = {MODEL_TYPE} ({model_name})")

# ── STARTUP DELAY (non-chief only) ────────────────────────────────────────────
if not is_chief:
    delay = args.start_delay if args.start_delay > 0 else 10
    log(f"[Startup]  Waiting {delay}s for chief to be ready…")
    time.sleep(delay)
    log(f"[Startup]  Done waiting")

# ── STRATEGY ──────────────────────────────────────────────────────────────────
log(f"\n[Strategy] Initialising MultiWorkerMirroredStrategy…")
strategy = tf.distribute.MultiWorkerMirroredStrategy()
log(f"[Strategy] replicas = {strategy.num_replicas_in_sync}")
log(f"[Worker]   type={task_type}  index={task_index}  is_chief={is_chief}")

# ── STEP 1: Load and split all-data.csv ───────────────────────────────────────
log("\n[Step 1] Loading dataset…")

if not os.path.isfile(DATA_PATH):
    log(f"[FATAL] CSV file not found at: {DATA_PATH}")
    log_file.close()
    sys.exit(1)

texts, labels = [], []
with open(DATA_PATH, encoding="latin-1") as f:
    for row in csv.reader(f):
        if len(row) >= 2:
            labels.append(LABEL_MAP.get(row[0].strip(), 1))
            texts.append(row[1].strip())

indices = list(range(len(texts)))
random.shuffle(indices)

split     = int(len(indices) * 0.8)
train_idx = indices[:split]
test_idx  = indices[split:]

train_texts  = [texts[i]  for i in train_idx]
train_labels = [labels[i] for i in train_idx]
test_texts   = [texts[i]  for i in test_idx]
test_labels  = [labels[i] for i in test_idx]

log(f"Total rows: {len(texts)}")
log(f"Train: {len(train_texts)} | Test: {len(test_texts)}")

# Fail loudly (and on every worker) rather than silently training on nothing.
if len(train_texts) == 0 or len(test_texts) == 0:
    log("[FATAL] Train or test set is empty after loading CSV — aborting before fit().")
    log(f"        train={len(train_texts)} test={len(test_texts)}  data_path={DATA_PATH}")
    log_file.close()
    sys.exit(1)

# ── STEP 2: TextVectorization ─────────────────────────────────────────────────
log("\n[Step 2] Adapting vectorizer…")
vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode="int",
    output_sequence_length=MAX_LEN,
    standardize="lower_and_strip_punctuation"
)
vectorizer.adapt(train_texts)
log(f"Vocabulary: {len(vectorizer.get_vocabulary())} tokens")

# Save the vocabulary (chief only) so prediction scripts can load it later
# without re-reading/re-processing the whole training CSV.
if is_chief:
    vocab_path = os.path.join(OUTPUT_DIR, "vocab.txt")
    with open(vocab_path, "w", encoding="utf-8") as f:
        for word in vectorizer.get_vocabulary():
            f.write(word + "\n")
    log(f"Vocabulary saved → {vocab_path}")

# ── STEP 3: tf.data pipelines ─────────────────────────────────────────────────
global_batch = BATCH_SIZE * strategy.num_replicas_in_sync
log(f"\n[Step 3] Global batch: {global_batch}  ({BATCH_SIZE} x {strategy.num_replicas_in_sync} replicas)")

def make_dataset(text_list, label_list, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((text_list, label_list))
    if shuffle:
        ds = ds.shuffle(len(text_list), seed=SEED)
    ds = ds.batch(global_batch).prefetch(tf.data.AUTOTUNE)
    return ds.map(lambda t, l: (vectorizer(t), l))

train_ds = make_dataset(train_texts, train_labels, shuffle=True)
test_ds  = make_dataset(test_texts,  test_labels,  shuffle=False)

# ── STEP 4: Build model inside strategy scope ─────────────────────────────────
log(f"\n[Step 4] Building {model_name} model…")

def build_model():
    if MODEL_TYPE == 1:
        rnn_layer = tf.keras.layers.SimpleRNN(32)
    else:
        rnn_layer = tf.keras.layers.GRU(32)

    return tf.keras.Sequential([
        tf.keras.layers.Embedding(VOCAB_SIZE, 32, input_length=MAX_LEN),
        tf.keras.layers.Bidirectional(rnn_layer),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(3, activation="softmax")
    ], name=model_name)

with strategy.scope():
    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

summary_path = os.path.join(OUTPUT_DIR, "model_summary.txt")
with open(summary_path, "w") as f:
    model.summary(print_fn=lambda x: f.write(x + "\n"))
log(f"Model summary → {summary_path}")

# ── STEP 5: Class weights + Train ─────────────────────────────────────────────
weights = compute_class_weight("balanced", classes=np.array([0, 1, 2]), y=train_labels)
class_weight = {0: weights[0], 1: weights[1], 2: weights[2]}
log(f"\nClass weights: negative={weights[0]:.2f}  neutral={weights[1]:.2f}  positive={weights[2]:.2f}")

ckpt_dir = os.path.join(OUTPUT_DIR, "checkpoints")
tb_dir   = os.path.join(OUTPUT_DIR, "tensorboard")
os.makedirs(ckpt_dir, exist_ok=True)

callbacks = [tf.keras.callbacks.TensorBoard(log_dir=tb_dir)]
if is_chief:
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(ckpt_dir, "ckpt_{epoch:02d}"),
        save_weights_only=True, verbose=1
    ))

log(f"\n[Step 5] Training {model_name}…")
t0 = time.time()
try:
    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        verbose=1 if is_chief else 0,
        class_weight=class_weight,
        callbacks=callbacks
    )
except Exception:
    log(f"\n[FATAL] model.fit() raised an exception on worker {task_index}:")
    log(traceback.format_exc())
    log_file.close()
    sys.exit(1)
train_time = time.time() - t0
log(f"\nTraining done in {train_time:.0f}s")

# ── Non-chief exits HERE — before any predict() calls ─────────────────────────
# Staying alive during model.predict() would trigger CollectiveReduceV2 ops
# across all workers, causing UnavailableError when other workers disconnect.
if not is_chief:
    log(f"\n[Worker {task_index}] Training complete — exiting cleanly.")
    log_file.close()
    sys.exit(0)

# ── STEP 6: Evaluate — chief only, fresh single-worker inference model ────────
try:
    log(f"\n[Step 6] Building single-worker inference model…")
    inference_model = build_model()
    inference_model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
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

    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0)
    acc    = accuracy_score(y_true, y_pred) * 100
    best   = max(history.history["accuracy"]) * 100

    report_path = os.path.join(OUTPUT_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"CLASSIFICATION REPORT  ({model_name} — MultiWorker, Kaggle CSV)\n")
        f.write("─" * 52 + "\n")
        f.write(report + "\n")
        f.write(f"Test accuracy : {acc:.2f}%\n")
        f.write(f"Best train acc: {best:.2f}%\n")
        f.write(f"Training time : {train_time:.0f}s\n")
        f.write(f"Parameters    : {model.count_params():,}\n")
        f.write(f"Replicas used : {strategy.num_replicas_in_sync}\n")
        f.write(f"Model type    : {MODEL_TYPE} ({model_name})\n")

    log("\n" + "─" * 52)
    log(f"CLASSIFICATION REPORT  ({model_name} — MultiWorker, Kaggle CSV)")
    log("─" * 52)
    log(report)
    log(f"Test accuracy : {acc:.2f}%")
    log(f"Best train acc: {best:.2f}%")
    log(f"Training time : {train_time:.0f}s")
    log(f"Parameters    : {model.count_params():,}")
    log(f"Replicas used : {strategy.num_replicas_in_sync}")
    log(f"Report saved  → {report_path}")

    # ── STEP 7: Sample predictions ────────────────────────────────────────────
    sentences = [
        "The company reported record profits and strong revenue growth",
        "Massive layoffs announced as sales decline sharply",
        "The firm maintained normal operations this quarter",
        "Shares tumble after the firm missed earnings expectations",
        "Investors cheered after the merger was finalized successfully",
    ]

    pred_path = os.path.join(OUTPUT_DIR, "sample_predictions.txt")
    log("\n" + "─" * 52)
    log(f"SAMPLE PREDICTIONS  ({model_name})")
    log("─" * 52)

    with open(pred_path, "w") as pf:
        pf.write(f"SAMPLE PREDICTIONS  ({model_name})\n" + "─" * 52 + "\n")
        for sentence in sentences:
            probs = inference_model.predict(vectorizer([sentence]), verbose=0)[0]
            idx   = probs.argmax()
            line  = (f"  Text  : {sentence}\n"
                     f"  Result: {CLASS_NAMES[idx].upper()}  ({probs[idx]*100:.0f}% confident)\n"
                     f"  Scores: neg={probs[0]:.2f}  neu={probs[1]:.2f}  pos={probs[2]:.2f}\n")
            log(line)
            pf.write(line + "\n")

    log(f"Predictions saved → {pred_path}")
    log(f"\n[Worker 0] All done. Outputs in: {OUTPUT_DIR}")
    log_file.close()
except Exception:
    log("\n[FATAL] Chief crashed during evaluation/report generation:")
    log(traceback.format_exc())
    log_file.close()
    sys.exit(1)
