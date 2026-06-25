### Simple RNN & GRU model container base (Docker)
We need to merge the two files from phase one. Merge the SimpleRNN and GRU for each Dataset (Kaggle & Twitter Financial news)
<center><img src="img/docker-container.png"></center>
<b>1-Kaggle for SimpleRNN and GRU</b>

<img src="img/kaggle-SimpleRNN-GRU.png">

<a href="birnn_bigru_kaggle_CIND860.py">birnn_bigru_kaggle_CIND860.py</a>

<b>2-Twitter financial news for SimpleRNN and GRU</b>

<img src="img/twitter-SimpleRNN-GRU.png">

<a href="birnn_bigru_twitter_financial_CIND860.py">birnn_bigru_twitter_financial_CIND860.py</a>

## Prepare the file for building images

<b>1- Dockerfile for Kaggle (SimpleRNN and GRU)</b> 

Dataset: Kaggle (file:all-data.csv)
Each image has:
<b>MODEL_TYPE</b>
MODEL_TYPE = 1  ->  Bidirectional SimpleRNN
MODEL_TYPE = 2  ->  Bidirectional GRU
<b>Input</b>
Input paramter: bitsimplernn-worker-(1|2|3 ...).env
This is to keep the setting configuration for the cluster
<b>Output</b>
Input volume point to input folder
output volume point to output folder


<b>2- Dockerfile for Twitter financial news (SimpleRNN and GRU)</b> 

Dataset: Kaggle (from Hugging face: zeroshot/twitter-financial-news-sentiment)
Each image has:
<b>MODEL_TYPE</b>
MODEL_TYPE = 1  ->  Bidirectional SimpleRNN
MODEL_TYPE = 2  ->  Bidirectional GRU
<b>Input</b>
Input paramter: bitsimplernn-worker-(1|2|3 ...).env
This is to keep the setting configuration for the cluster
<b>Output</b>
Input volume point to input folder
output volume point to output folder

## Build the image with Docker file


