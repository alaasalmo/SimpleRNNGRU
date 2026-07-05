# Data Analysis

## I. Choosing the Epochs methodology
According the checking for the percentage of the two data sets, we see this data is moderate imbalance. For this reason, when we want to specifify the best epoch, we will depend on two factors (Accuracy + Macro-F1). Combining them — (accuracy + macro_f1) / 2 — gives you a single number that rewards a model for both being generally correct and not neglecting the minority class. That's the correct instinct given the moderate imbalance you found (4.3:1 and 4.8:1 ratios).

```
Twitter Financial News (0=bearish/negative, 1=bullish/positive, 2=neutral):
Label index Class name   Count%   total
0     bearish(negative)  1,789    15%
1     bullish (positive) 2.398    20%
2     neutral            7,744    64.9
```

```
Kaggke Financial:
Label Classname Count   Percentage
0     negative  604     12.46%
1     neutra    l2,879  59.4
2     positive  1,363   28.13%
```

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


## II. Choosing the best Hyperparameters by using the loops and find the best epochs
For the two data set (Twits financial-news & Kaggle)

Choose the best Hyperparameters from the array: 

```
1- Learning Rate: [0.0005, 0.001, 0.002, 0.005]
2- Vocab Size [6000, 10000, 15000]
3- Embedding Dimension [32, 64, 128]
4- RNN Units[32, 64, 128]
```

<a href="hyperparameter">hyperparameter</a>

## III. Choose the best models (SimppleRNN or DiSimpleRNN and GRU or DiGRU)
We use the Hyperparameters for step II.

<a href="/modelcomparision">model-comparision</a>

## IV. Choose Dropout percentage for each models
    
<a href="dropout">dropout</a>

## V. Building full model:
 
```
Augmentation -> GloVe -> BiSimpleRNN -> Attention -> Dense -> Softmax
Augmentation -> GloVe -> BiGRU -> Attention -> Dense -> Softmax
```

<a href="pipeline">pipeline</a>
