# SimpleRNNGRU

## Twitter Financial News (Hagging face) 

The dataset contains finance-related tweets and news-style short texts labeled by sentiment.

Dataset size: ~12,000 financial tweets/text samples

Label: 0 = bearish 1 = bullish 2 = neutral

Refrence: https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment?utm_source=chatgpt.com

<a href="simple_rnn_twitter_financial_kaggle_CIND860.py">simple_rnn_twitter_financial_kaggle_CIND860.py</a>

<b>Result:</b>

```
────────────────────────────────────────────────────

CLASSIFICATION REPORT  (SimpleRNN)

────────────────────────────────────────────────────
              precision    recall  f1-score   support

     bearish       0.00      0.00      0.00       347
     bullish       0.40      0.46      0.43       475  
	 neutral       0.76      0.90      0.82      1566
    accuracy                           0.68      2388
   macro avg       0.39      0.45      0.42      2388
weighted avg       0.58      0.68      0.63      2388

Test accuracy : 67.96%
Best train acc: 82.26%
Training time : 32s
Parameters    : 394,435
```

<a href="gru_twitter_financial_kaggle_CIND860.py">gru_twitter_financial_kaggle_CIND860.py</a>

<b>Result:</b>

```
────────────────────────────────────────────────────
CLASSIFICATION REPORT  (GRU)
────────────────────────────────────────────────────
              precision    recall  f1-score   support

     bearish       0.00      0.00      0.00       347
     bullish       0.00      0.00      0.00       475
     neutral       0.66      1.00      0.79      1566
    accuracy                           0.66      2388
   macro avg       0.22      0.33      0.26      2388
weighted avg       0.43      0.66      0.52      2388

Test accuracy : 65.58%
Best train acc: 64.74%
Training time : 75s
Parameters    : 411,139

```
## Sentiment Analysis For Financial News (kaggle)

The Dataset contains Financial PhraseBank dataset created by Malo et al. and contains financial news headlines labeled according to how a retail investor would perceive their impact on the market.

Label: 0 = Positive 1 = Negative 2 = Neutral

Refernce: https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news 

<a href="simple_rnn_kaggle_CIND860.py">simple_rnn_kaggle_CIND860.py</a>

<b>Result:</b>

```

────────────────────────────────────────────────────
CLASSIFICATION REPORT (SimpleRNN)
──────────────────────────────────────────────────
              precision    recall  f1-score   support

    negative       0.44      0.42      0.43       105
     neutral       0.75      0.78      0.77       585
    positive       0.56      0.53      0.54       280
    accuracy                           0.67       970
   macro avg       0.58      0.58      0.58       970
weighted avg       0.66      0.67      0.67       970


```

<a href="gru_kaggle_CIND860.py">gru_kaggle_CIND860.py</a>

<b>Result:</b>

```
────────────────────────────────────────────────────
CLASSIFICATION REPORT  (GRU)
────────────────────────────────────────────────────
              precision    recall  f1-score   support

    negative       0.00      0.00      0.00       105
     neutral       0.82      0.86      0.84       585
    positive       0.53      0.66      0.58       280

    accuracy                           0.71       970
   macro avg       0.45      0.51      0.47       970
weighted avg       0.64      0.71      0.68       970

```

## Improved Sentiment Analysis For Financial News (kaggle)

1- Adding Bidirectional: It's more expressive but it runs the sequence forward and backward, then concatenates both outputs, doubling the context the model sees for each word.

2- Adding Dropout: Overfitting is a major issue on small financial sentiment datasets

3- Add Class weight when we have unbalance result for the class. calculates weights inversely proportional to frequency

###SimpleRNN

<a href="simpleRNN_improved_CIN860.py">simpleRNN_improved_CIN860.py</a>

<b>Result:</b>

```

────────────────────────────────────────────────────
CLASSIFICATION REPORT  (SimpleRNN)
────────────────────────────────────────────────────
              precision    recall  f1-score   support

    negative       0.35      0.43      0.38       105
     neutral       0.74      0.75      0.75       585
    positive       0.55      0.48      0.51       280

    accuracy                           0.64       970
   macro avg       0.54      0.55      0.55       970
weighted avg       0.64      0.64      0.64       970

```

###GRU


<b>Result:</b>

```
────────────────────────────────────────────────────
CLASSIFICATION REPORT  (BiGRU)
────────────────────────────────────────────────────
              precision    recall  f1-score   support

    negative       0.58      0.65      0.61       105
     neutral       0.82      0.79      0.80       585
    positive       0.65      0.66      0.65       280

    accuracy                           0.74       970
   macro avg       0.68      0.70      0.69       970
weighted avg       0.74      0.74      0.74       970
```