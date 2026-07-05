#Implement the pipeline 

FINAL RESULTS — Augmentation -> GloVe -> BiSimpleRNN -> Attention -> Dense -> Softmax
FINAL RESULTS — Augmentation -> GloVe -> BiGRU -> Attention -> Dense -> Softmax

## BiSimpleRNN for Twitter financial
Use:
```
First:
SimpleGRU | dropout=0.1

Second:
Best Learning Rate [0.0005, 0.001, 0.002, 0.005]: 0.0005
Best Vocab Size [6000, 10000, 15000]: 6000
Best Embedding Dimension [32, 64, 128]: 64
Best RNN Units [32, 64, 128]: 64
```

Output:
```
============================================================
FINAL RESULTS — Augmentation -> GloVe -> BiSimpleRNN -> Attention -> Dense -> Softmax
============================================================
Hyperparameters used: {'learning_rate': 0.0005, 'vocab_size': 6000, 'embedding_dim': 100, 'rnn_units': 64, 'dropout_rate': 0.1, 'max_len': 60}
Test Accuracy   : 79.27%
Test Macro-F1   : 0.7014
Test Combined   : 0.7470
Bearish F1: 0.582 | Bullish F1: 0.646 | Neutral F1: 0.876
Best epoch      : 9 (out of 14 trained)
```

<a href="glove_augmentation_bisimplernn_attention_twitter.py">glove_augmentation_bisimplernn_attention_twitter.py</a>

## BiGRU for Twitter financial
Use:
```
First:
BiGRU | dropout=0.2

Second:
Best Learning Rate [0.0005, 0.001, 0.002, 0.005]: 0.002
Best Vocab Size [6000, 10000, 15000]: 15000
Best Embedding Dimension [32, 64, 128]: 128
Best RNN Units [32, 64, 128]: 128
```

Output:
```
============================================================
FINAL RESULTS — Augmentation -> GloVe -> BiGRU -> Attention -> Dense -> Softmax
============================================================
Hyperparameters used: {'learning_rate': 0.0005, 'vocab_size': 6000, 'embedding_dim': 100, 'rnn_units': 64, 'dropout_rate': 0.1, 'max_len': 60}
Test Accuracy   : 81.62%
Test Macro-F1   : 0.7392
Test Combined   : 0.7777
Bearish F1: 0.634 | Bullish F1: 0.697 | Neutral F1: 0.887
Best epoch      : 12 (out of 17 trained)
```

<a href="glove_augmentation_bigru_attention_twitter.py">glove_augmentation_bigru_attention_twitter.py</a>

## BiSimpleRNN for Kaggle financial

Use:
```
First:
BiSimpleRNN | dropout=0.2

Second:
Best Learning Rate [0.0005, 0.001, 0.002, 0.005]: 0.0005
Best Vocab Size [6000, 10000, 15000]: 6000
Best Embedding Dimension [32, 64, 128]: 128
Best RNN Units [32, 64, 128]: 32
```

Output:
```
============================================================
FINAL RESULTS — Augmentation -> GloVe -> BiSimpleRNN -> Attention -> Dense -> Softmax
============================================================
Hyperparameters used: {'learning_rate': 0.0005, 'vocab_size': 6000, 'embedding_dim': 100, 'rnn_units': 32, 'dropout_rate': 0.2, 'max_len': 60}
Test Accuracy   : 75.52%
Test Macro-F1   : 0.7161
Test Combined   : 0.7357
Negative F1: 0.705 | Neutral F1: 0.822 | Positive F1: 0.621
Best epoch      : 13 (out of 18 trained)
```

<a href="glove_augmentation_bisimplernn_attention_kaggle.py">glove_augmentation_bisimplernn_attention_kaggle.py</a>

## BiGRU for Kaggle financial

Use:

```
First:
BiGRU | dropout=0.2
Second:
Best Learning Rate [0.0005, 0.001, 0.002, 0.005]: 0.0005
Best Vocab Size [6000, 10000, 15000]: 6000
Best Embedding Dimension [32, 64, 128]: 128
Best RNN Units [32, 64, 128]: 32
```

Output:
```
============================================================
FINAL RESULTS — Augmentation -> GloVe -> BiGRU -> Attention -> Dense -> Softmax
============================================================
Hyperparameters used: {'learning_rate': 0.0005, 'vocab_size': 6000, 'embedding_dim': 100, 'rnn_units': 32, 'dropout_rate': 0.2, 'max_len': 60}
Test Accuracy   : 77.99%
Test Macro-F1   : 0.7380
Test Combined   : 0.7590
Negative F1: 0.721 | Neutral F1: 0.846 | Positive F1: 0.647
Best epoch      : 13 (out of 18 trained)
```

<a href="glove_augmentation_bigru_attention_kaggle.py">glove_augmentation_bigru_attention_kaggle.py</a>