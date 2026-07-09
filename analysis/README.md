# Data Analysis and Result 

This section explains the methodology used to select the optimal number of training epochs and presents the best hyperparameters for each model. It then compares the performance of the SimpleRNN and Bidirectional SimpleRNN models, followed by a comparison between the GRU and Bidirectional GRU models. Finally, it presents the overall project pipeline and summarizes the corresponding results.

## I. Choosing the Epochs methodology

Based on the class distribution analysis of the two datasets, both datasets exhibit a moderate class imbalance. Therefore, when selecting the optimal number of training epochs, we consider two evaluation metrics: accuracy and Macro-F1 score. We combine these metrics using the following equation:

`Model Score = (Accuracy + Macro-F1) / 2`	​

This combined score provides a balanced evaluation by rewarding models that achieve high overall accuracy while also maintaining good performance across all classes, including the minority class. Given the moderate imbalance observed in the datasets (class ratios of 4.3:1 and 4.8:1), this approach provides a more reliable criterion for selecting the best-performing model than relying on accuracy alone.

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

## III. Choose the best models (SimppleRNN or BiSimpleRNN and GRU or BiGRU)
We use the Hyperparameters for step II.

<a href="modelcomparision">model-comparision</a>

## IV. Choose Dropout percentage for each models
    
<a href="dropout">dropout</a>

## V. Building full model & result:
 
```
Augmentation -> GloVe -> BiSimpleRNN -> Attention -> Dense -> Softmax
Augmentation -> GloVe -> BiGRU -> Attention -> Dense -> Softmax
```

<a href="pipeline">Model pipeline and final result</a>
