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

### Model: SimpleRNN

============================================================
Best SimpleRNN Hyperparameters Found
============================================================
Best Learning Rate [0.0005, 0.001, 0.002, 0.005]: 0.0005
Best Vocab Size [6000, 10000, 15000]: 6000
Best Embedding Dimension [32, 64, 128]: 32
Best RNN Units [32, 64, 128]: 32

{'learning_rate': 0.0005, 'vocab_size': 6000, 'embedding_dim': 32, 'rnn_units': 32, 'max_len': 60, 'dropout_rate': 0.45}

<a href="simplernn_hyperparameter_sweep.py">simplernn_hyperparameter_sweep.py</a>

### Model: BiSimpleRNN

============================================================
Best BiSimpleRNN Hyperparameters Found
============================================================
Best Learning Rate [0.0005, 0.001, 0.002, 0.005]: 0.0005
Best Vocab Size [6000, 10000, 15000]: 6000
Best Embedding Dimension [32, 64, 128]: 64
Best RNN Units [32, 64, 128]: 32


{'learning_rat': 0.0005, 'vocab_size': 6000, 'embedding_dim': 64, 'rnn_units': 32, 'max_len': 60, 'dropout_rate': 0.45}

<a href="bisimplernn_hyperparameter_sweep.py">bisimplernn_hyperparameter_sweep.py</a>

### Model: GRU

============================================================
Best BiGRU Hyperparameters Found
============================================================


Best Learning Rate [0.0005, 0.001, 0.002]: 0.0005
Best Vocab Size [6000, 10000, 15000]: 6000
Best Embedding Dimension [32, 64, 128]: 128
Best GRU Units [32, 64, 128]: 32
=== FINAL BEST HYPERPARAMETER COMBINATION ===

{'learning_rate': 0.0005, 'vocab_size': 6000, 'embedding_dim': 32, 'rnn_units': 32, 'max_len': 60, 'dropout_rate': 0.45}

<a href="gru_hyperparameter_sweep.py">gru_hyperparameter_sweep.py</a>

### Model: BiGRU

============================================================
Best BiGRU Hyperparameters Found
============================================================

Best Learning Rate [0.0005, 0.001, 0.002, 0.005]: 0.005
Best Vocab Size [6000, 10000, 15000]: 6000
Best Embedding Dimension [32, 64, 128]: 128
Best BiGRU Units [32, 64, 128]: 32

{'learning_rate': 0.005, 'vocab_size': 6000, 'embedding_dim': 128, 'rnn_units': 32, 'max_len': 60, 'dropout_rate': 0.45}

<a href="bigru_hyperparameter_sweep.py">bigru_hyperparameter_sweep.py</a>