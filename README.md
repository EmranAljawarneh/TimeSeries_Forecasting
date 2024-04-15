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

![Screenshot (8601)](https://github.com/EmranAljawarneh/TimeSeriesForecasting/assets/44652088/30be8cef-69ee-4fe5-bda1-17571abff631)
