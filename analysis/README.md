# Analysis for Data (Twitter financial New)

## Calculate the methods to choose the best Hyperparameters

Best Learning Rate [0.0005, 0.001, 0.002]:

Best Vocab Size [6000, 10000, 15000]

Best Embedding Dimension [32, 64, 128]

Best Vocab Size [6000, 10000, 15000]



We calculated for each class

Precision=TP /(FP+TP)

Recall=TP/(FN+TP)​

F1= (Precision + Recall)/2

Macro-F1=(F1 bearish​ + F1 bullish​ + F1 neutral​​)/3

How to use:

Loop by value depending Macro-F1 (the best value)

Calculate Macro-F1:

We calculated for each class

Precision=TP /(FP+TP)

Recall=TP/(FN+TP)​

F1= (Precision + Recall)/2

Macro-F1=(F1 bearish​ + F1 bullish​ + F1 neutral​​)/3

## Best Hyperparameters (Twitter financial New)

### Model: BiSimpleRNN

```
Best BiSimpleRNN Hyperparameters Found
Best Learning Rate [0.0005, 0.001, 0.002, 0.005]: 0.0005
Best Vocab Size [6000, 10000, 15000]: 6000
Best Embedding Dimension [32, 64, 128]: 32
Best RNN Units [32, 64, 128]: 32
Final combined BiSimpleRNN -> Accuracy: 77.43%  Macro-F1: 0.6820
Dataset label distribution

Train class counts -> Bearish: 1442 (15.1%) | Bullish: 1923 (20.2%) | Neutral: 6178 (64.7%)
Test class counts -> Bearish: 347 (14.5%) | Bullish: 475 (19.9%) | Neutral: 1566 (65.6%)
```

<a href="bisimplernn_hyperparameter_twitter.py">bisimplernn_hyperparameter_twitter.py</a>

```
EXPERIMENT 4 SUMMARY
  SimpleRNN       best_macro_f1=0.8417  best_epoch=20
  BiSimpleRNN     best_macro_f1=0.9237  best_epoch=13

FINAL OVERALL SUMMARY

Best dropout setting (Exp 1):     dropout=0.45
Best class-weight setting:        OFF
Best augmentation setting (Exp 3): ON
SimpleRNN best macro-F1:     0.8417 (epoch 20)
BiSimpleRNN best macro-F1:   0.9237 (epoch 13)
```
<b>SimpleRNN (dropout, class weight, Augmentation, Bi-directional</b>

Dropout:

<img src="img/dropout_comparison_simplernn_twitter.png">

Class weight:

<img src="img/class_weight_comparison_simplernn_twitter.png">

Augmentation (depending on NLP)

<img src="img/augmentation_comparison_simplernn_twitter.png">

SimpleRNN vs BiSimpleRNN

<img src="img/simplernn_vs_bisimplernn_comparison_simplernn_twitter.png">

<a href="four_experiments_simplernn_twitter.py">four_experiments_simplernn_twitter.py</a>


### Model: BiGRU

<b>MENTION — Best BiGRU Hyperparameters Found</b>

Best Learning Rate [0.0005, 0.001, 0.002, 0.005]: 0.005

Best Vocab Size [6000, 10000, 15000]: 6000

Best Embedding Dimension [32, 64, 128]: 128

Best RNN Units [32, 64, 128]: 128

Final combined BiGRU -> Accuracy: 81.95%  Macro-F1: 0.7387

Train class counts -> Bearish: 1442 (15.1%) | Bullish: 1923 (20.2%) | Neutral: 6178 (64.7%)

Test class counts -> Bearish: 347 (14.5%) | Bullish: 475 (19.9%) | Neutral: 1566 (65.6%)

<a href="bigru_hyperparameter_twitter.py">bigru_hyperparameter_twitter.py</a>

<b>GRU (dropout, class weight, Augmentation, Bi-directional)</b>

Best dropout setting (Exp 1): dropout=0.45

```
EXPERIMENT 4 SUMMARY
  GRU             best_macro_f1=0.2620  best_epoch=1
  BiGRU           best_macro_f1=0.7109  best_epoch=5

FINAL OVERALL SUMMARY

Best dropout setting (Exp 1):     no dropout
Best class-weight setting:        OFF
Best augmentation setting (Exp 3): OFF
GRU best macro-F1:     0.2620 (epoch 1)
BiGRU best macro-F1:   0.7109 (epoch 5)
```

Dropout:

<img src="img/dropout_comparison_gru_twitter.png">

Class weight:

<img src="img/class_weight_comparison_gru_twitter.png">

Augmentation (depending on NLP)

<img src="img/augmentation_comparison_gru_twitter.png">

SimpleRNN vs BiSimpleRNN

<img src="img/gru_vs_bigru_comparison_gru_twitter.png">


<a href="four_experiments_gru_twitter.py">four_experiments_gru_twitter.py</a>

# Analysis for Data (Kaggle)

## Model: BiSimpleRNN

### Calculate the methods to choose the best Hyperparameters

```
MENTION — Best BiSimpleRNN Hyperparameters Found
Best Learning Rate [0.0005, 0.001, 0.002, 0.005]: 0.0005
Best Vocab Size [6000, 10000, 15000]: 6000
Best Embedding Dimension [32, 64, 128]: 64
Best RNN Units [32, 64, 128]: 32
Final combined BiSimpleRNN -> Accuracy: 68.23%  Macro-F1: 0.6335
Dataset label distribution
Train class counts -> Negative: 513 (12.5%) | Neutral: 2447 (59.4%) | Positive: 1159 (28.1%)
Test class counts -> Negative: 91 (12.5%) | Neutral: 432 (59.4%) | Positive: 204 (28.1%)
```

<a href="bisimplernn_hyperparameter_kaggle.py">bisimplernn_hyperparameter_kaggle.py</a>

```
FINAL OVERALL SUMMARY
SimpleRNN       best_macro_f1=0.5691  best_epoch=16
BiSimpleRNN     best_macro_f1=0.6353  best_epoch=6
Best dropout setting (Exp 1): dropout=0.45
Best class-weight setting:    OFF
Best augmentation setting:    ON
SimpleRNN best macro-F1:     0.5691 (epoch 16)
BiSimpleRNN best macro-F1:   0.6353 (epoch 6)
```

<b>SimpleRNN (dropout, class weight, Augmentation, Bi-directional</b>

Dropout:

<img src="img/dropout_comparison_simplernn_kaggle.png">

Class weight:

<img src="img/class_weight_comparison_simplernn_kaggle.png">

Augmentation (depending on NLP)

<img src="img/augmentation_comparison_simplernn_kaggle.png">

SimpleRNN vs BiSimpleRNN

<img src="img/simplernn_vs_bisimplernn_comparison_simplernn_kaggle.png">

<a href="four_experiments_simplernn_kaggle.py">four_experiments_simplernn_kaggle.py</a>

## Model: GRU

### Calculate the methods to choose the best Hyperparameters

```MENTION — Best BiSimpleRNN Hyperparameters Found

Best Learning Rate [0.0005, 0.001, 0.002, 0.005]: 0.0005
Best Vocab Size [6000, 10000, 15000]: 6000
Best Embedding Dimension [32, 64, 128]: 64
Best RNN Units [32, 64, 128]: 32

Final combined BiSimpleRNN -> Accuracy: 68.23%  Macro-F1: 0.6335

Dataset label distribution
Train class counts -> Negative: 513 (12.5%) | Neutral: 2447 (59.4%) | Positive: 1159 (28.1%)
Test class counts -> Negative: 91 (12.5%) | Neutral: 432 (59.4%) | Positive: 204 (28.1%)
```
<a href="bisimplernn_hyperparameter_kaggle.py">bisimplernn_hyperparameter_kaggle.py</a>

```
EXPERIMENT 4 SUMMARY
  GRU             best_macro_f1=0.6314  best_epoch=12
  BiGRU           best_macro_f1=0.5907  best_epoch=8

=============================================
FINAL OVERALL SUMMARY
=============================================
Best dropout setting (Exp 1): no dropout
Best class-weight setting:    OFF
Best augmentation setting:    ON
GRU best macro-F1:     0.6314 (epoch 12)
BiGRU best macro-F1:   0.5907 (epoch 8)

```

<b>SimpleRNN (dropout, class weight, Augmentation, Bi-directional</b>

Dropout:

<img src="img/dropout_comparison_gru_kaggle.png">

Class weight:

<img src="img/class_weight_comparison_gru_kaggle.png">

Augmentation (depending on NLP)

<img src="img/augmentation_comparison_gru_kaggle.png">

SimpleRNN vs BiSimpleRNN

<img src="img/gru_vs_bigru_comparison_gru_kaggle.png">

<a href="four_experiments_gru_kaggle.py">four_experiments_gru_kaggle.py</a>