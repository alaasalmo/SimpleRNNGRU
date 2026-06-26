# =====================================================
#  Sentiment Analysis with SimpleRNN
#  For Beginners — Step by Step
#  Dataset: all-data.csv (Financial News)
# =====================================================
#  Install: pip install tensorflow scikit-learn
#  Run:     python3 simple_rnn_sentiment.py
# =====================================================

import csv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# ── STEP 1: Load the CSV file ─────────────────────
texts  = []
labels = []

with open("all-data.csv", encoding="latin-1") as f:
    for row in csv.reader(f):
        if len(row) >= 2:
            labels.append(row[0].strip())   # negative / neutral / positive
            texts.append(row[1].strip())    # the sentence

print(f"✓ Loaded {len(texts)} sentences")

# ── STEP 2: Turn labels into numbers ──────────────
#   negative → 0,  neutral → 1,  positive → 2
encoder = LabelEncoder()
y = encoder.fit_transform(labels)
print(f"✓ Classes: {list(encoder.classes_)}")

# ── STEP 3: Turn words into numbers ───────────────
#   "profit" → 42,  "loss" → 87,  etc.
VOCAB_SIZE = 5000   # only keep the 5000 most common words
MAX_LEN    = 50     # use the first 50 words of each sentence

tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<unknown>")
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)          # words → numbers
X = pad_sequences(sequences, maxlen=MAX_LEN, padding="post")  # same length

print(f"Each sentence is now {X.shape[1]} numbers long")

# ── STEP 4: Split into training and test sets ──────
#   80% for training,  20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training: {len(X_train)} sentences | Test: {len(X_test)} sentences")

# ── STEP 5: Build the SimpleRNN model ─────────────
#
#   Embedding  →  turns word numbers into vectors
#   SimpleRNN  →  reads the sentence word by word
#   Dense      →  gives the final answer (3 classes)
#
model = Sequential([
    Embedding(input_dim=VOCAB_SIZE, output_dim=32),   # word → vector
    SimpleRNN(32),                                     # read the sequence
    Dense(3, activation="softmax")                    # output: 3 classes
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ── STEP 6: Train the model ────────────────────────
print("\n📚 Training...")
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=64,
    validation_data=(X_test, y_test),
    verbose=1
)

# ── STEP 7: Check results ──────────────────────────
print("\n📊 Results on test data:")
y_pred = model.predict(X_test, verbose=0).argmax(axis=1)

print(classification_report(
    y_test, y_pred,
    target_names=encoder.classes_,
    zero_division=0
))

# ── STEP 8: Try your own sentence ─────────────────
def predict(sentence):
    seq = tokenizer.texts_to_sequences([sentence])
    seq = pad_sequences(seq, maxlen=MAX_LEN, padding="post")
    probs = model.predict(seq, verbose=0)[0]
    idx   = probs.argmax()
    label = encoder.classes_[idx]
    conf  = probs[idx] * 100
    print(f"  '{sentence}'")
    print(f"  → {label.upper()}  ({conf:.0f}% confident)\n")

print("🔍 Predictions:")
predict("The company reported record profits and strong growth")
predict("Massive layoffs announced as sales decline")
predict("The firm maintained normal operations this quarter")
