# Stock Price Prediction using PCA and LSTM

This repository contains Python scripts for predicting stock prices using Long Short-Term Memory (LSTM) networks. The project includes scripts for Google stock price prediction using LSTM.

## Google Stock Price Prediction with LSTM
- Dataset: Google Stock Price dataset containing historical stock prices.
- Features: The opening stock prices.
- Technique: Long Short-Term Memory (LSTM) network is used for time series prediction.
- Training: The dataset is split into training and testing sets. The training set consists of the 60 previous stock prices, and the testing set contains the next day's stock price.
- Model Architecture: The LSTM model consists of four LSTM layers with dropout regularization, followed by a dense output layer.
