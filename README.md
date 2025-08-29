# QCoin implied volatility forecasting

This repo contains my Kaggle notebook work for short horizon implied volatility prediction for QCoin using classic time series features and recurrent neural networks with attention

## goals

1. build strong baselines that are easy to run inside Kaggle
2. add robust checkpointing so training can resume after any disconnect
3. compare several recurrent models on the same data splits
4. document findings with clear metrics and next steps

## data and task

Target is ten second ahead implied volatility for QCoin  
Features come from the prepared train and test tensors in the notebook  
Evaluation uses mean squared error root mean squared error mean absolute error and R squared

## models

All models use the same dataset API batch loader and loss  
Each model has an attention pooling layer after the recurrent stack

* GRU with attention
* BiGRU with attention
* LSTM with attention
* BiLSTM with attention
* deep stacked LSTM with attention five layers
* deep stacked BiLSTM with attention five layers

## training setup

* optimizer AdamW with weight decay
* loss mean squared error
* batch size one two eight
* epochs fifty
* device CUDA when available else CPU

## robust checkpoints and auto resume

Kaggle can end a session if the notebook is idle  
To avoid loss the notebook now saves torch checkpoints every ten epochs and at the final epoch and automatically resumes from the newest file

Where files go
* checkpoints  
  * torch_AttentionGRU  
    * epoch_10.pt epoch_20.pt epoch_30.pt epoch_40.pt epoch_50.pt  
  * torch_AttentionBiGRU  
    * epoch_10.pt …

What is inside each file
* current epoch number
* model state dict
* optimizer state dict

How resume works
* when train_model starts it scans the model checkpoint folder
* it loads the newest state and sets start_epoch to that value
* the loop continues from start_epoch to EPOCHS with no extra action needed

## current results

All runs use the same split and metric computation

GRU with attention  
MSE 2.138222  
RMSE 1.462266  
MAE 0.645799  
R squared 0.161661

BiGRU with attention  
MSE 2.366343  
RMSE 1.538292  
MAE 0.704461  
R squared 0.072220

LSTM with attention  
MSE 2.256068  
RMSE 1.502021  
MAE 0.717862  
R squared 0.115457

BiLSTM with attention  
MSE 2.495132  
RMSE 1.579599  
MAE 0.689045  
R squared 0.021726

deep stacked LSTM with attention five layers  
MSE 2.208426  
RMSE 1.486077  
MAE 0.661851  
R squared 0.134136

deep stacked BiLSTM with attention five layers  
training in progress at the time of writing

## quick analysis

Summary  
* GRU with attention currently leads on both error and R squared  
* deep stacked LSTM with attention is a close second  
* bidirectional variants did not improve on this split and show higher error which suggests either over capacity or suboptimal regularization for this dataset size and horizon

Interpretation  
* attention on top of a single direction recurrent encoder seems sufficient for this horizon and feature set  
* the modest gap between GRU and deep stacked LSTM indicates that deeper temporal mixing helps but may need stronger regularization to avoid noise fitting
