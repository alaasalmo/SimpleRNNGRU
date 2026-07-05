# Find the best dropout and choose best Epoch

## BiSimpleRNN for Twitter financial

```
--- SimpleGRU | dropout=0.1 ---
  Epoch 1: val_acc=0.7179 val_f1=0.4533 combined=0.5856
  Epoch 2: val_acc=0.7668 val_f1=0.5892 combined=0.6780
  Epoch 3: val_acc=0.7870 val_f1=0.6796 combined=0.7333
  Epoch 4: val_acc=0.7835 val_f1=0.6760 combined=0.7298
  Epoch 5: val_acc=0.7842 val_f1=0.6828 combined=0.7335
  Epoch 6: val_acc=0.7772 val_f1=0.6829 combined=0.7301
  Epoch 7: val_acc=0.7842 val_f1=0.6842 combined=0.7342
  Epoch 8: val_acc=0.7814 val_f1=0.6831 combined=0.7323
  Epoch 9: val_acc=0.7898 val_f1=0.6823 combined=0.7361
  Epoch 10: val_acc=0.7856 val_f1=0.6882 combined=0.7369
  Epoch 11: val_acc=0.7814 val_f1=0.6804 combined=0.7309
  Epoch 12: val_acc=0.7814 val_f1=0.6656 combined=0.7235
  Epoch 13: val_acc=0.7647 val_f1=0.6737 combined=0.7192
  Early stopping at epoch 13 (no improvement for 3 epochs)
Final (stopped epoch 13) | test_acc=0.7990 test_f1=0.7065 combined=0.7527
```

<img src="img\bisimplernn_dropout_comparison_twitter.png">

<a href="bisimplernn_dropout_sweep_twitter.py"></a>

## BiGRU for Twitter financial

```
--- BiGRU | dropout=0.2 ---
  Epoch 1: val_acc=0.7291 val_f1=0.4623 combined=0.5957
  Epoch 2: val_acc=0.7535 val_f1=0.5724 combined=0.6630
  Epoch 3: val_acc=0.8003 val_f1=0.7143 combined=0.7573
  Epoch 4: val_acc=0.7905 val_f1=0.6896 combined=0.7401
  Epoch 5: val_acc=0.7800 val_f1=0.7054 combined=0.7427
  Epoch 6: val_acc=0.7856 val_f1=0.7072 combined=0.7464
  Early stopping at epoch 6 (no improvement for 3 epochs)
Final (stopped epoch 6) | test_acc=0.8137 test_f1=0.7326 combined=0.7731
```

<img src="img\bigru_dropout_comparison_twitter.png">

<a href="bigru_dropout_sweep_twitter.py">bigru_dropout_sweep_twitter.py</a>

## BiSimpleRNN for Kaggle financial

```
--- BiSimpleRNN | dropout=0.2 ---
  Epoch 1: val_acc=0.6731 val_f1=0.4253 combined=0.5492
  Epoch 2: val_acc=0.6958 val_f1=0.4632 combined=0.5795
  Epoch 3: val_acc=0.7039 val_f1=0.5666 combined=0.6352
  Epoch 4: val_acc=0.6958 val_f1=0.5687 combined=0.6322
  Epoch 5: val_acc=0.6699 val_f1=0.5418 combined=0.6059
  Epoch 6: val_acc=0.6845 val_f1=0.5761 combined=0.6303
  Epoch 7: val_acc=0.6796 val_f1=0.5642 combined=0.6219
  Epoch 8: val_acc=0.6715 val_f1=0.5202 combined=0.5959
  Epoch 9: val_acc=0.6748 val_f1=0.5531 combined=0.6139
  Epoch 10: val_acc=0.6667 val_f1=0.5519 combined=0.6093
  Epoch 11: val_acc=0.6796 val_f1=0.5676 combined=0.6236
  Epoch 12: val_acc=0.6780 val_f1=0.5586 combined=0.6183
Final (epoch 12) | test_acc=0.6850 test_f1=0.5971 combined=0.6411
```

<img src="img\bisimplernn_dropout_comparison_kaggle.png">

<a href="bisimplernn_dropout_sweep_kaggle.py">bisimplernn_dropout_sweep_kaggle.py</a>

## BiGRU for Kaggle financial

```
--- BiGRU | dropout=0.2 ---
  Epoch 1: val_acc=0.6877 val_f1=0.4464 combined=0.5670
  Epoch 2: val_acc=0.6942 val_f1=0.4969 combined=0.5955
  Epoch 3: val_acc=0.7540 val_f1=0.6950 combined=0.7245
  Epoch 4: val_acc=0.7508 val_f1=0.6976 combined=0.7242
  Epoch 5: val_acc=0.7201 val_f1=0.6830 combined=0.7015
  Epoch 6: val_acc=0.7460 val_f1=0.6936 combined=0.7198
  Early stopping at epoch 6 (no improvement for 3 epochs)
Final (stopped epoch 6) | test_acc=0.7455 test_f1=0.6902 combined=0.7179
```

<img src="img\bigru_dropout_comparison_kaggle.png">

<a href="bigru_dropout_sweep_kaggle.py">bigru_dropout_sweep_kaggle.py</a>

