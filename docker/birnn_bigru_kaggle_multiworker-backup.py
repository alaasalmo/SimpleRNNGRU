#  BiRNN / BiGRU — Financial Sentiment Analysis (Kaggle all-data.csv)
#  Multi-Worker Mirrored Strategy
#
#  Usage:
#    python birnn_bigru_kaggle_multiworker.py --input /data/input --output /data/output --model-type 1
#
#  MODEL_TYPE:
#    1 = Bidirectional SimpleRNN
#    2 = Bidirectional GRU
#
#  TF_CONFIG is injected via environment variable (env-file per worker).

import os, sys, csv, json, random, time, argparse
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight

# ── ARGS ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="BiRNN/BiGRU MultiWorker Training (Kaggle)")
parser.add_argument("--input",       required=True, help="Input directory containing all-data.csv")
parser.add_argument("--output",      required=True, help="Output directory (logs, checkpoints, report)")
parser.add_argument("--model-type",  type=int, default=1, choices=[1, 2],
                    help="1 = Bidirectional SimpleRNN, 2 = Bidirectional GRU (default: 1)")
parser.add_argument("--start-delay", type=int, default=10,
                    help="Seconds non-chief workers wait before connecting (default: 10)")
args = parser.parse_args()

INPUT_DIR   = args.input
OUTPUT_DIR  = args.output
MODEL_TYPE  = args.model_type
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_PATH = os.path.join(INPUT_DIR, "all-data.csv")

# ── CONFIG ────────────────────────────────────────────────────────────────────
BATCH_SIZE = 64
VOCAB_SIZE = 5000
MAX_LEN    = 50
EPOCHS     = 10
LABEL_MAP  = {"negative": 0, "neutral": 1, "positive": 2}
CLASS_NAMES = ["negative", "neutral", "positive"]

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# ── READ TF_CONFIG EARLY ──────────────────────────────────────────────────────
tf_config  = json.loads(os.environ.get("TF_CONFIG", "{}"))
task_type  = tf_config.get("task", {}).get("type", "worker")
task_index = tf_config.get("task", {}).get("index", 0)
is_chief   = (task_type == "worker" and task_index == 0)

# ── LOGGING ───────────────────────────────────────────────────────────────────
log_path = os.path.join(OUTPUT_DIR, "train.log")
log_file = open(log_path, "w", buffering=1)

def log(msg=""):
    print(msg, flush=True)
    log_file.write(msg + "\n")

model_name = "BiRNN" if MODEL_TYPE == 1 else "BiGRU"

log(f"{'='*52}")
log(f"{model_name}  |  worker-{task_index}  ({'chief' if is_chief else 'worker'})")
log(f"{'='*52}")
log(f"[Config]   input      = {INPUT_DIR}")
log(f"[Config]   output     = {OUTPUT_DIR}")
log(f"[Config]   data_path  = {DATA_PATH}")
log(f"[Config]   model_type = {MODEL_TYPE} ({model_name})")

if not os.path.isfile(DATA_PATH):
    log(f"ERROR: {DATA_PATH} not found. Mount the Kaggle CSV into the input volume.")
    log_file.close()
    sys.exit(1)

# ── STARTUP DELAY (non-chief only) ────────────────────────────────────────────
if not is_chief:
    delay = args.start_delay if args.start_delay > 0 else 10
    log(f"[Startup]  Waiting {delay}s for chief to be ready…")
    time.sleep(delay)
    log(f"[Startup]  Done waiting ✓")

# ── STRATEGY ──────────────────────────────────────────────────────────────────
log(f"\n[Strategy] Initialising MultiWorkerMirroredStrategy…")
strategy = tf.distribute.MultiWorkerMirroredStrategy()
log(f"[Strategy] replicas = {strategy.num_replicas_in_sync}")
log(f"[Worker]   type={task_type}  index={task_index}  is_chief={is_chief}")

# ── STEP 1: Read CSV + split 80/20 ────────────────────────────────────────────
log("\n[Step 1] Reading CSV and splitting 80/20…")
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

N_TRAIN = len(train_texts)
N_TEST  = len(test_texts)
log(f"Total rows   : {len(texts)}")
log(f"Training rows: {N_TRAIN}  |  Test rows: {N_TEST}")

# ── STEP 2: tf.data pipelines (scaled for multi-worker) ───────────────────────
global_batch     = BATCH_SIZE * strategy.num_replicas_in_sync
STEPS_PER_EPOCH  = N_TRAIN // global_batch
VALIDATION_STEPS = N_TEST  // global_batch + 1
log(f"\n[Step 2] Global batch: {global_batch}  ({BATCH_SIZE} x {strategy.num_replicas_in_sync} replicas)")
log(f"Steps/epoch: {STEPS_PER_EPOCH}  |  Validation steps: {VALIDATION_STEPS}")

def make_dataset(text_list, label_list, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices((text_list, label_list))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=N_TRAIN, seed=42)
    dataset = dataset.batch(global_batch)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

train_ds = make_dataset(train_texts, train_labels, shuffle=True)
test_ds  = make_dataset(test_texts,  test_labels,  shuffle=False)
log("tf.data pipelines ready")

# ── STEP 3: TextVectorization ─────────────────────────────────────────────────
log("\n[Step 3] Adapting vectorizer…")
vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode="int",
    output_sequence_length=MAX_LEN
)
vectorizer.adapt(train_texts)
log(f"Vocabulary built: {len(vectorizer.get_vocabulary())} tokens")

def vectorize(text, label):
    return vectorizer(text), label

train_ds = train_ds.map(vectorize)
test_ds  = test_ds.map(vectorize)

# ── STEP 4: Build model inside strategy scope ─────────────────────────────────
log(f"\n[Step 4] Building {model_name} model…")

def build_model():
    if MODEL_TYPE == 1:
        rnn_layer = tf.keras.layers.SimpleRNN(32)
    else:
        rnn_layer = tf.keras.layers.GRU(32, recurrent_dropout=0.2)

    return tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=VOCAB_SIZE, output_dim=32, input_length=MAX_LEN),
        tf.keras.layers.Bidirectional(rnn_layer),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(3, activation="softmax")
    ], name=model_name)

with strategy.scope():
    model = build_model()
    model.compile(
        optimizer="adam",
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
history = model.fit(
    train_ds,
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=EPOCHS,
    validation_data=test_ds,
    validation_steps=VALIDATION_STEPS,
    class_weight=class_weight,
    verbose=1 if is_chief else 0,
    callbacks=callbacks
)
train_time = time.time() - t0
log(f"\nTraining done in {train_time:.0f}s")

# ── Non-chief exits HERE — before any predict() calls ─────────────────────────
if not is_chief:
    log(f"\n[Worker {task_index}] Training complete — exiting cleanly.")
    log_file.close()
    sys.exit(0)

# ── STEP 6: Evaluate — chief only, fresh single-worker inference model ────────
log(f"\n[Step 6] Building single-worker inference model…")
inference_model = build_model()
inference_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
inference_model.set_weights(model.get_weights())
log("Weights transferred ✓")

test_ds_infer = (
    tf.data.Dataset.from_tensor_slices((test_texts, test_labels))
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
    .map(vectorize)
)

log("Evaluating…")
y_true, y_pred = [], []
for text_batch, label_batch in test_ds_infer:
    preds = inference_model.predict(text_batch, verbose=0).argmax(axis=1)
    y_pred.extend(preds.tolist())
    y_true.extend(label_batch.numpy().tolist())

y_true = y_true[:N_TEST]
y_pred = y_pred[:N_TEST]

report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0)

report_path = os.path.join(OUTPUT_DIR, "classification_report.txt")
with open(report_path, "w") as f:
    f.write(f"CLASSIFICATION REPORT  ({model_name} — MultiWorker)\n")
    f.write("─" * 52 + "\n")
    f.write(report + "\n")
    f.write(f"Training time : {train_time:.0f}s\n")
    f.write(f"Parameters    : {model.count_params():,}\n")
    f.write(f"Replicas used : {strategy.num_replicas_in_sync}\n")
    f.write(f"Model type    : {MODEL_TYPE} ({model_name})\n")

log("\n" + "─" * 52)
log(f"CLASSIFICATION REPORT  ({model_name} — MultiWorker)")
log("─" * 52)
log(report)
log(f"Training time : {train_time:.0f}s")
log(f"Parameters    : {model.count_params():,}")
log(f"Replicas used : {strategy.num_replicas_in_sync}")
log(f"Report saved  → {report_path}")

# ── STEP 7: Sample predictions ────────────────────────────────────────────────
sentences = [
    "The company reported record profits and strong revenue growth",
    "Massive layoffs announced as sales decline sharply",
    "The firm maintained normal operations this quarter",
]

pred_path = os.path.join(OUTPUT_DIR, "sample_predictions.txt")
log("\n" + "─" * 52)
log(f"SAMPLE PREDICTIONS  ({model_name})")
log("─" * 52)

with open(pred_path, "w") as pf:
    pf.write(f"SAMPLE PREDICTIONS  ({model_name})\n" + "─" * 52 + "\n")
    for sentence in sentences:
        vec   = vectorizer([sentence])
        probs = inference_model.predict(vec, verbose=0)[0]
        idx   = probs.argmax()
        label = CLASS_NAMES[idx]
        conf  = probs[idx] * 100
        line  = f"  Text  : {sentence}\n  Result: {label.upper()}  ({conf:.0f}% confident)\n"
        log(line)
        pf.write(line + "\n")

log(f"Predictions saved → {pred_path}")
log(f"\n[Worker 0] All done. Outputs in: {OUTPUT_DIR}")
log_file.close()
