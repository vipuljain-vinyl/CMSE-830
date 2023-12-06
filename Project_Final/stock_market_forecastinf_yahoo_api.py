import requests
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from streamlit_searchbox import st_searchbox
import yfinance as yf
import altair as alt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.api import SARIMAX




# Page setup
st.set_page_config(page_title="Stock Price Prediction", layout="wide")
st.title("Stock Price Prediction")

stock_list_df= pd.read_csv('nasdaq_screener.csv')

# Use a text_input to get the keywords to filter the dataframe
searchterm = st.selectbox("Search stock by symbol or company name", options=list(stock_list_df[['Symbol','Name']].apply(tuple,axis=1)), index = None)
    
st.write('You selected:', searchterm)

stock_ticker=None
duration = "6mo"
interval = "1d"

if searchterm:
    stock_ticker = searchterm[0]
    stock = searchterm[1]
    st.write(' stock_ticker:', stock_ticker)

    st.header(stock, divider='rainbow')

if stock_ticker:
    try:
        stock_data = yf.download(stock_ticker, period=duration, interval=interval)
        # stock_data = stock_data.sort_values(by="Date",ascending=False)
        st.write('_Historical_ ',duration,' _Data at_',interval,'_-interval:_')
        
        st.write("Latest Date ",stock_data.tail(1).index.values)
        st.write("Oldest Date ",stock_data.head(1).index.values)
        st.write(stock_data.head(10))
    except Exception as e:
        st.write(f'Error fetching data: {e}')
    
    st.header(f"Exploratory Data Analysis for {searchterm[1]}")
    # st.line_chart(stock_data['Close'])
    # st.bar_chart(stock_data[['Open', 'High', 'Low', 'Close']])

    tab1, tab2, tab3 = st.tabs(["Historical Movements", "Seasonality & trend", "Insights"])

    with tab1:
        st.subheader(f"Open, High, Low, Close for {stock}")
        bar_chart = alt.Chart(stock_data.reset_index(),title="Evolution of stock prices").transform_fold(
            ['Open', 'High', 'Low', 'Close'],
            as_=['Variable', 'Value']
        ).mark_line().encode(
            x='Date:T',
            y='Value:Q',
            color='Variable:N',
            tooltip=['Date:T', 'Value:Q']
        ).properties(
            width=800,
            height=400
        )

        st.altair_chart(bar_chart, use_container_width=True)

        st.title("Data Summary")
        st.dataframe(stock_data.describe())

        st.title("Data Summary")
        st.write(stock_data.reset_index()['Date'].min(),stock_data.reset_index()['Date'].max())
        selected_date = st.date_input("Select Date",value = None, min_value=stock_data.reset_index()['Date'].min(), max_value=stock_data.reset_index()['Date'].max())
        selected_row = stock_data.reset_index()[stock_data.reset_index()['Date'] == str(selected_date)]
        st.write(f"Selected Date: {selected_date}")
        st.write(selected_row)

    with tab2:
        # Decompose time series into trend, seasonality, and residual components
        result = seasonal_decompose(stock_data['Close'], model='additive', period=7)

        # Plot components
        st.title(f"Seasonality Decomposition for {stock}")
        st.subheader("Original Close Prices")
        st.line_chart(stock_data['Close'])

        st.subheader("Trend Component")
        st.line_chart(result.trend.dropna())

        st.subheader("Seasonal Component")
        st.line_chart(result.seasonal.dropna())

        st.subheader("Residual Component")
        st.line_chart(result.resid.dropna())

        # Plot the combined trend and seasonality
        fig, ax = plt.subplots(figsize=(10, 6))
        result.trend.dropna().plot(ax=ax, label='Trend')
        result.seasonal.dropna().plot(ax=ax, label='Seasonality')
        ax.legend()
        ax.set_title("Combined Trend and Seasonality")
        st.pyplot(fig)

    with tab3:
        st.title("Correlation Heatmap")
        corr_matrix = stock_data.corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        st.pyplot(fig)

        st.title("Explore the distribution of individual features.")
        feature_to_plot = st.selectbox("Select Feature for Distribution", stock_data.columns)
        st.title(f"Distribution of {feature_to_plot}")
        fig, ax = plt.subplots()
        sns.histplot(stock_data[feature_to_plot], bins=30, kde=True, ax = ax)
        st.pyplot(fig)

        st.title("Custom Date Range Analysis")
        start_date = st.date_input("Select Start Date",value = None, min_value=stock_data.reset_index()['Date'].min(), max_value=stock_data.reset_index()['Date'].max())
        end_date = st.date_input("Select End Date",value = None, min_value=stock_data.reset_index()['Date'].min(), max_value=stock_data.reset_index()['Date'].max())
        st.write(start_date,end_date) ###
        if start_date and end_date:
            selected_data = stock_data.reset_index()[(stock_data.reset_index()['Date'] >= str(start_date)) & (stock_data.reset_index()['Date'] <= str(end_date))]
            st.dataframe(selected_data)


        st.header(" ")
        window_size = st.slider("Select Moving Average Window Size", min_value=1, max_value=30, value=7)
        st.title(f"Closing Price with {window_size}-day Moving Average")
        stock_data_temp1 = stock_data.copy()
        stock_data_temp1['MA'] = stock_data_temp1['Close'].rolling(window=window_size).mean()
        st.line_chart(stock_data_temp1[['Close', 'MA']])

    

    st.title("Forecasting Stock Price")
    tab21, tab22, tab23, tab24 = st.tabs([" ExponentialSmoothing ", " Moving Average - SARIMA ", " Exponential Moving Average "," LSTM "])
    

    test_ratio = 0.2
    training_ratio = 1 - test_ratio

    train_size = int(training_ratio * len(stock_data))
    test_size = int(test_ratio * len(stock_data))
    

    train = stock_data[:train_size][["Close"]]
    test = stock_data[train_size:][["Close"]]

    ## Split the time-series data into training seq X and output value Y
    def extract_seqX_outcomeY(data, N, offset):
        """
        Split time-series into training sequence X and outcome value Y
        Args:
            data - dataset
            N - window size, e.g., 50 for 50 days of historical stock prices
            offset - position to start the split
        """
        X, y = [], []

        for i in range(offset, len(data)):
            X.append(data[i - N : i])
            y.append(data[i])

        return np.array(X), np.array(y)


    #### Calculate the metrics RMSE and MAPE ####
    def calculate_rmse(y_true, y_pred):
        """
        Calculate the Root Mean Squared Error (RMSE)
        """
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        return rmse


    def calculate_mape(y_true, y_pred):
        """
        Calculate the Mean Absolute Percentage Error (MAPE) %
        """
        y_pred, y_true = np.array(y_pred), np.array(y_true)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        return mape

    ############################################
    def calculate_perf_metrics(var, stockdata_):
        ### RMSE
        rmse = calculate_rmse(
            np.array(stockdata_[train_size:]["Close"]),
            np.array(stockdata_[train_size:][var]),
        )
        ### MAPE
        mape = calculate_mape(
            np.array(stockdata_[train_size:]["Close"]),
            np.array(stockdata_[train_size:][var]),
        )

        return rmse, mape
    
    def plot_stock_trend(var, cur_title, stockprices):
        fig, ax = plt.subplots(figsize=(10, 6))
        stockprices[["Close", var, "100day"]].plot(ax=ax)
        plt.grid(False)
        plt.title(cur_title)
        plt.axis("tight")
        plt.ylabel("Stock Price ($)")
        st.pyplot(fig)

    with tab21:
        model = ExponentialSmoothing(stock_data['Close'], seasonal='add', seasonal_periods=7)
        fitted_model = model.fit()
        forecast = fitted_model.forecast(steps=1)

        st.write(f"Forecast for the next day's closing value: {forecast.iloc[0]}")

    with tab22:
        window_size = st.slider("Select Moving Average Window Size", min_value=1, max_value=50, value=7)
        window_var = f"{window_size}day"

        stock_data_MA = stock_data.copy()
        stock_data_MA[window_var] = stock_data_MA["Close"].rolling(window_size).mean()

        ### Include a 100-day SMA for reference
        stock_data_MA["100day"] = stock_data_MA["Close"].rolling(100).mean()


        ### Plot and performance metrics for SMA model
        plot_stock_trend(var=window_var, cur_title="Simple Moving Averages", stockprices = stock_data_MA)



        rmse_sma, mape_sma = calculate_perf_metrics(var=window_var  , stockdata_ = stock_data_MA)

        st.write(" RMSE for MA ", rmse_sma, " MAPE for MA ",mape_sma)

        # Create the SARIMA object
        #sarma = SARIMAXe(stock_data_MA['Close'], window=20)





    """
    


    
    """

        


    









