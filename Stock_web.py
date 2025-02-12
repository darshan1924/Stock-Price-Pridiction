import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
 
st.title("Stock Price Predictor App")
 
# Stock ticker input from user 
stock = st.text_input("Enter the Stock ID", "GOOG")
 

# Define start and end dates for fetching stock data (last 20 years)
from datetime import datetime
end = datetime.now()
start = datetime(end.year-20,end.month,end.day)
google_data = yf.download(stock, start, end)
 

# Load the pre-trained stock prediction model
model = load_model("Latest_stock_price_model.keras")


# Display the latest 20 records from the dataset
st.subheader("Stock Data")
st.write(google_data.sort_index(ascending=False).head(20))


# Define data split for training and testing (70% train, 30% test)
splitting_len = int(len(google_data)*0.7)
x_test = google_data[['Close']].iloc[splitting_len:].copy()


# Function to plot moving averages and stock prices
def plot_graph(figsize, values, full_data, extra_data = 0, extra_dataset = None):
    fig = plt.figure(figsize=figsize)
    plt.plot(values,'Orange')
    plt.plot(full_data.Close, 'b')
    if extra_data:
        plt.plot(extra_dataset)
    return fig
 

# Compute and visualize Moving Averages (MA) for different periods
st.subheader('Original Close Price and MA for 250 days')
google_data['MA_for_250_days'] = google_data.Close.rolling(250).mean()
st.pyplot(plot_graph((15,6), google_data['MA_for_250_days'],google_data,0))
 
st.subheader('Original Close Price and MA for 200 days')
google_data['MA_for_200_days'] = google_data.Close.rolling(200).mean()
st.pyplot(plot_graph((15,6), google_data['MA_for_200_days'],google_data,0))
 
st.subheader('Original Close Price and MA for 100 days')
google_data['MA_for_100_days'] = google_data.Close.rolling(100).mean()
st.pyplot(plot_graph((15,6), google_data['MA_for_100_days'],google_data,0))
 
st.subheader('Original Close Price and MA for 100 days and MA for 250 days')
st.pyplot(plot_graph((15,6), google_data['MA_for_100_days'],google_data,1,google_data['MA_for_250_days']))
 
from sklearn.preprocessing import MinMaxScaler
 

# Scale the test data for model input (Normalize values between 0 and 1)
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(x_test)  
 

# Prepare input sequences for model testing (100-day window size) 
x_data = []
y_data = []
for i in range(100,len(scaled_data)):
    x_data.append(scaled_data[i-100:i])
    y_data.append(scaled_data[i])
 
x_data, y_data = np.array(x_data), np.array(y_data)
 

# Generate predictions on test data
predictions = model.predict(x_data)
 

# Convert predictions back to actual stock price values 
inv_pre = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_data)
 

# Create a DataFrame to compare actual vs predicted values 
ploting_data = pd.DataFrame(
{
  'original_test_data': inv_y_test.reshape(-1),
    'predictions': inv_pre.reshape(-1)
} ,
    index = google_data.index[splitting_len+100:]
)


# Display the Original and Predicted data
st.subheader("Original values vs Predicted values")
st.write(ploting_data.sort_index(ascending=False).head(20))
 

# Plot the graph between original and predicted data
st.subheader('Original Close Price vs Predicted Close price')
fig = plt.figure(figsize=(15,6))
plt.plot(pd.concat([google_data.Close[:splitting_len+100],ploting_data], axis=0))
plt.legend(["Data- not used", "Original Test data", "Predicted Test data"])
st.pyplot(fig)


# ----------- Future Stock Price Prediction -----------

st.markdown(
    "<h3 style='color: Red; font-size: 24px;'>⚠️ Prediction Alert!</h3>"
    "<p style='color: white; font-size: 18px;'>These predictions are based on historical trends and don't guarantee future performance. "
    "Stock prices are highly volatile and subject to market conditions.</p>",
    unsafe_allow_html=True
)

future_days = 10

# Take the last 100 days from the existing data as input
last_100_days = google_data[['Close']].iloc[-100:].values
scaled_last_100_days = scaler.transform(last_100_days)

future_predictions = []

for _ in range(future_days):
    input_data = np.array(scaled_last_100_days[-100:]).reshape(1, 100, 1)
    next_day_scaled = model.predict(input_data)
    next_day_price = scaler.inverse_transform(next_day_scaled)[0][0]
    future_predictions.append(next_day_price)
    scaled_last_100_days = np.append(scaled_last_100_days, next_day_scaled, axis=0)

# Create DataFrame for future predictions
future_dates = pd.date_range(start=google_data.index[-1], periods=future_days+1, freq='B')[1:] 
future_df = pd.DataFrame({'Future Predictions': future_predictions}, index=future_dates)

# Display future predictions
st.subheader(f"Predicted Stock Prices for the Next {future_days} Days")
st.write(future_df)

# Plot the future predictions
fig_future = plt.figure(figsize=(15, 6))
plt.plot(google_data.index[-100:], google_data['Close'].iloc[-100:], label="Last 100 Days Actual Prices", color='blue')
plt.plot(future_df.index, future_df['Future Predictions'], label="Future Predictions", color='red', linestyle='dashed')
plt.legend()
st.pyplot(fig_future)
