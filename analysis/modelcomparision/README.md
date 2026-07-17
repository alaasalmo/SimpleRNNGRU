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


```
======================================================================

SimpleRNN  (Acc: 59.38%  Macro-F1: 0.4536)
------------------------------------------
                 Bearish         Bullish         Neutral          Total      
True: Bearish  106 (30.5%)      83 (23.9%)     158 (45.5%)         347       
True: Bullish   95 (20.0%)     152 (32.0%)     228 (48.0%)         475       
True: Neutral  203 (13.0%)     203 (13.0%)     1160 (74.1%)        1566      

BiSimpleRNN  (Acc: 78.64%  Macro-F1: 0.6801)
--------------------------------------------
                 Bearish         Bullish         Neutral          Total      
True: Bearish  164 (47.3%)      49 (14.1%)     134 (38.6%)         347       
True: Bullish   47 (9.9%)      275 (57.9%)     153 (32.2%)         475       
True: Neutral   58 (3.7%)       69 (4.4%)      1439 (91.9%)        1566    

Better model (by combined score): BiSimpleRNN (combined=0.7333, best epoch=6)

```

<a href="simplernn_vs_bisimplernn_twitter.py">simplernn_vs_bisimplernn_twitter.py</a>

## Compare GRU and BiGRU for Twitter financial 

```
Model              Acc  Macro-F1  Combined  Best Epoch   Stopped
----------------------------------------------------------------
GRU             65.58%    0.2640    0.4599           1         6
BiGRU           81.07%    0.7142    0.7625           2         7
```

```
Confusion Matrices (row-normalized) — Test Set
======================================================================

GRU  (Acc: 65.58%  Macro-F1: 0.2640)
------------------------------------
                 Bearish         Bullish         Neutral          Total      
True: Bearish    0 (0.0%)        0 (0.0%)      347 (100.0%)        347       
True: Bullish    0 (0.0%)        0 (0.0%)      475 (100.0%)        475       
True: Neutral    0 (0.0%)        0 (0.0%)     1566 (100.0%)        1566      

BiGRU  (Acc: 81.16%  Macro-F1: 0.7126)
--------------------------------------
                 Bearish         Bullish         Neutral          Total      
True: Bearish  166 (47.8%)      57 (16.4%)     124 (35.7%)         347       
True: Bullish   35 (7.4%)      304 (64.0%)     136 (28.6%)         475       
True: Neutral   32 (2.0%)       66 (4.2%)      1468 (93.7%)        1566      

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

```
Confusion Matrices (row-normalized) — Test Set
======================================================================

SimpleRNN  (Acc: 63.69%  Macro-F1: 0.3897)
------------------------------------------
                  Negative        Neutral         Positive         Total      
True: Negative    0 (0.0%)       60 (65.9%)      31 (34.1%)          91       
True: Neutral     0 (0.0%)      396 (91.7%)      36 (8.3%)          432       
True: Positive    0 (0.0%)      137 (67.2%)      67 (32.8%)         204       

BiSimpleRNN  (Acc: 66.02%  Macro-F1: 0.4835)
--------------------------------------------
                  Negative        Neutral         Positive         Total      
True: Negative   10 (11.0%)      31 (34.1%)      50 (54.9%)          91       
True: Neutral     1 (0.2%)      375 (86.8%)      56 (13.0%)         432       
True: Positive    7 (3.4%)      102 (50.0%)      95 (46.6%)         204       
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
