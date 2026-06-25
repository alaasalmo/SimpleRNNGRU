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

We need to build the image through building Dockerfile

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

-Input volume point to input folder

-output volume point to output folder

File requirements for Dockerfile.kaggle

- <a href="requirements.kaggle.txt">requirements.kaggle.txt</a>
- <a href="birnn_bigru_kaggle_multiworker.py">birnn_bigru_kaggle_multiworker.py</a>


```
# =============================================================================
#  MODEL_TYPE is chosen at *runtime* via --model-type 1|2, not baked into
#  the image, so the same image serves both BiRNN and BiGRU workers.
#
#  Volumes:
#    /data/input   (ro)  → must contain all-data.csv
#    /data/output  (rw)  → train.log, classification_report.txt,
#                          sample_predictions.txt, checkpoints/, tensorboard/
# =============================================================================
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY birnn_bigru_kaggle_multiworker.py .

VOLUME ["/data/input", "/data/output"]
EXPOSE 12345

# --model-type, --input, --output, --start-delay are passed at `docker run` time
ENTRYPOINT ["python", "birnn_bigru_kaggle_multiworker.py"]
```


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

-Input volume point to input folder

-output volume point to output folder

## Build the image with Docker file


