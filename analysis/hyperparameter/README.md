# Find the best hyperparamters and choose best Epoch


## BiSimpleRNN for Twitter financial

```
============================================================
Best BiSimpleRNN Hyperparameters Found
============================================================
Best Learning Rate [0.0005, 0.001, 0.002, 0.005]: 0.0005
Best Vocab Size [6000, 10000, 15000]: 6000
Best Embedding Dimension [32, 64, 128]: 64
Best RNN Units [32, 64, 128]: 64

Final combined BiSimpleRNN -> Accuracy: 79.40%  Macro-F1: 0.6883
Best epoch (FINAL config): 9

```

<a href="bisimplernn_hyperparameter_twitter.py">bisimplernn_hyperparameter_twitter.py</a>

## BiGRU for Twitter financial 

```
============================================================
Best BiGRU Hyperparameters Found
============================================================
Best Learning Rate [0.0005, 0.001, 0.002, 0.005]: 0.002
Best Vocab Size [6000, 10000, 15000]: 15000
Best Embedding Dimension [32, 64, 128]: 128
Best RNN Units [32, 64, 128]: 128

Final combined BiGRU -> Accuracy: 78.56%  Macro-F1: 0.7095
Best epoch (FINAL config): 3

------------------------------------------------------------
Dataset label distribution
------------------------------------------------------------
Train class counts -> Bearish: 1442 (15.1%) | Bullish: 1923 (20.2%) | Neutral: 6178 (64.7%)
Test class counts -> Bearish: 347 (14.5%) | Bullish: 475 (19.9%) | Neutral: 1566 (65.6%)
```

<a href="bigru_hyperparameter_twitter.py">bigru_hyperparameter_twitter.py</a>

## BiSimpleRNN for Kaggle financial

```
============================================================
Best BiSimpleRNN Hyperparameters Found
============================================================
Best Learning Rate [0.0005, 0.001, 0.002, 0.005]: 0.0005
Best Vocab Size [6000, 10000, 15000]: 6000
Best Embedding Dimension [32, 64, 128]: 128
Best RNN Units [32, 64, 128]: 32

Final combined BiSimpleRNN -> Accuracy: 66.99%  Macro-F1: 0.5448
Best epoch (FINAL config): 15

------------------------------------------------------------
Dataset label distribution
------------------------------------------------------------
Train class counts -> Negative: 513 (12.5%) | Neutral: 2447 (59.4%) | Positive: 1159 (28.1%)
Test class counts -> Negative: 91 (12.5%) | Neutral: 432 (59.4%) | Positive: 204 (28.1%)
```

<a href="bisimplernn_hyperparameter_kaggle.py">bisimplernn_hyperparameter_kaggle.py</a> 

## BiGRU for Kaggle financial

```
============================================================
Best BiSimpleRNN Hyperparameters Found
============================================================
Best Learning Rate [0.0005, 0.001, 0.002, 0.005]: 0.0005
Best Vocab Size [6000, 10000, 15000]: 6000
Best Embedding Dimension [32, 64, 128]: 128
Best RNN Units [32, 64, 128]: 32

Final combined BiSimpleRNN -> Accuracy: 66.99%  Macro-F1: 0.5448
Best epoch (FINAL config): 15

------------------------------------------------------------
Dataset label distribution
------------------------------------------------------------
Train class counts -> Negative: 513 (12.5%) | Neutral: 2447 (59.4%) | Positive: 1159 (28.1%)
Test class counts -> Negative: 91 (12.5%) | Neutral: 432 (59.4%) | Positive: 204 (28.1%)
```

<a href="bigru_hyperparameter_kaggle.py">bigru_hyperparameter_kaggle.py</a>

