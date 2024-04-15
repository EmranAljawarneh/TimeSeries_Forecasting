# Time Series Forecasting Model for the Stock Market

Time series data prediction is an essential area of research in finance, and 
economics, among others. It involves analyzing and modeling data collected over 
time to make future predictions or forecast future trends. With the increasing 
availability of historical data and advancements in machine learning and deep 
learning techniques, time series data prediction has become an increasingly 
popular research topic in recent years. In this work, we investigate the application 
of machine and deep learning models to time-series data for predicting the optimal 
time for trading stocks and options. Time-series data is defined as a collection of 
historical data points ordered by time, commonly used in predicting stock prices, 
stock indexes, and cryptocurrency prices. We use a publicly available 
dataset of Japanese stocks and options to train and test Support Vector Regression 
(SVR) and Long Short-Term Memory (LSTM) models. The goal is 
to improve trading strategies by identifying the best times to buy and sell assets 
based on predictive models. The performance of the model is compared using 
three accuracy measurements: Mean Squared Error (MSE), Root Mean Squared 
Error (RMSE), and Mean Absolute Error (MAE). The study has shown that The 
LSTM with dropout technique provided the best possible results with MSE 
0.000124763, RMSE 0.011169727, and MAE 0.009058733.

# Dataset
The time-series data is considered numerical data points. Financial data is an example of 
this type. It consists of numeric variables. Time series data is usually fractional numbers 
(also known as decimal numbers), which takes any value between two numbers, known as 
continuous data. This data contains historical data for a variety of Japanese stocks and 
options. The data used in this study were obtained from the Kaggle website for data 
scientists and machine learning practitioners. The dataset was created by the Japan 
Exchange Group, which operates one of the largest stock exchanges in the world. It 
contains historic data for various Japanese stocks and options starting from 4/1/2017 and 
ending in 27/5/2022.

The dataset we used shown in this link: https://www.kaggle.com/competitions/jpx-tokyo-stock-exchange-prediction/data


# Stationary Time Series Test
Each time series dataset should be tested to determine if it exhibits stationarity, meaning 
that its statistical properties such as mean and variance are constant over time. Summary 
statistics are one of several methods used to assess the non-stationarity of a time series. 
This involves splitting the time series into two or more partitions and comparing the mean 
and variance of each group. If the means and variances differ significantly 
between the groups, the time series is likely non-stationary.

![Screenshot (8602)](https://github.com/EmranAljawarneh/TimeSeriesForecasting/assets/44652088/5176946d-8fb3-4f19-bf5b-a2ba836e9269) A big difference between the mean and variance as shown in the original data 
indicates that the data is non-stationary. To make it stationary, we applied the natural 
logarithm to the data. Fig. 2 and Fig. 3 present the behavior of the data before and after the 
stationary test. Fig. 3 shows the suitable format that is accepted by the model.
![Screenshot (8605)](https://github.com/EmranAljawarneh/TimeSeriesForecasting/assets/44652088/1d4945d1-06cd-4625-a230-eabd6532db92) ![Screenshot (8603)](https://github.com/EmranAljawarneh/TimeSeriesForecasting/assets/44652088/65fca8ac-78eb-44b9-81e1-75fe256c68e8)

# SVM Experiments
To identify the optimal hyperparameters for the Support Vector Regression (SVR) 
model. Various experiments were conducted to determine the optimal statistical model 
with the lowest error. The values of ”C” and ”ϵ” were investigated by experimenting with 
different values of ”C” and ”ϵ” while the kernel function remained fixed at the RBF 
function. Based on the evaluation metrics we got, it was found that ”C” = 
100 and,”ϵ” = 0.0005 provided the smallest error. To confirm that these values 
were indeed optimal, additional experiments were conducted with different values of ”C” 
while ”ϵ” was fixed at 0.0005.

# LSTM Experiments
Two experiments were conducted in this study, one with and one without the dropout
technique, using different numbers of neurons, namely 4, 45, 50, 55, 123, 128, and 133. 
The LSTM model with dropout technique outperformed the LSTM without dropout, as 
evidenced by the lower MSE, RMSE, and MAE metrics with values of 0.000124763, 
0.011169727, and 0.009058733. A comparison between the 
The LSTM model outperformed the SVR model by producing the 
best results, with an MSE of 0.000124763, an RMSE of 0.011169727, and an MAE of 
0.009058733.
