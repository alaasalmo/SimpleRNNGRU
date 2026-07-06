# Compare between regular model and BiDirection model and choose best Epoch
We will depend in two items:
1- Default paramters:
2- When we choose the best epoch we will depend on the claculation for Accuracy and Macro-F1.
   Using Macro-F1 when we have unbalance labels

We used Hyperparameters as default only to compare the models: 
{'learning_rate': 0.001, 'vocab_size': 6000, 'embedding_dim': 64, 'rnn_units': 64, 'max_len': 60}   

## Compare SimpleRNN and BiSimpleRNN for Twitter financial 

``` 
Model              Acc  Macro-F1  Combined  Best Epoch   Stopped
----------------------------------------------------------------
SimpleRNN       59.38%    0.4536    0.5237          18        23
BiSimpleRNN     78.64%    0.6801    0.7333           6        11
```
<img src="img\simplernn_vs_bisimplernn_twitter.png">

<a href="simplernn_vs_bisimplernn_twitter.py">simplernn_vs_bisimplernn_twitter.py</a>

## Compare GRU and BiGRU for Twitter financial 

```
Model              Acc  Macro-F1  Combined  Best Epoch   Stopped
----------------------------------------------------------------
GRU             65.58%    0.2640    0.4599           1         6
BiGRU           81.07%    0.7142    0.7625           2         7
```
<img src="img\gru_vs_bigru_twitter.png">

<a href="gru_vs_bigru_twitter.py">gru_vs_bigru_twitter.py</a>

## Compare SimpleRNN and BiSimpleRNN for Kaggle financial 

```
Model              Acc  Macro-F1  Combined  Best Epoch   Stopped
----------------------------------------------------------------
SimpleRNN       63.69%    0.3897    0.5133           2         7
BiSimpleRNN     66.02%    0.4835    0.5719           2         7

Better model (by combined score): BiSimpleRNN (combined=0.5719, best epoch=2)
```
<img src="img\simplernn_vs_bisimplernn_kaggle.png">

<a href="simplernn_vs_bisimplernn_kaggle.py">simplernn_vs_bisimplernn_kaggle.py</a>

## Compare GRU and BiGRU for Kaggle financial 


```
Model              Acc  Macro-F1  Combined  Best Epoch   Stopped
----------------------------------------------------------------
GRU             67.68%    0.6195    0.6481          18        23
BiGRU           72.90%    0.6809    0.7050           8        13

Better model (by combined score): BiGRU (combined=0.7050, best epoch=8)
```

<img src="img\gru_vs_bigru_kaggle.png">

<a href="gru_vs_bigru_kaggle.py">gru_vs_bigru_kaggle.py</a>
