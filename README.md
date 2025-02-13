### 📈 Stock Price Predictor App

### 🚀 Overview
The Stock Price Predictor App is a machine learning-powered application designed to analyze historical stock data, visualize trends, and predict future stock prices. Built with Streamlit, it leverages Yahoo Finance for real-time data retrieval and a Keras-based LSTM model for accurate short-term predictions. This app bridges the gap between financial analysis and machine learning, offering users a seamless experience for stock market insights.

### 📌 Key Features
Real-Time Stock Data: Fetch live stock data using the Yahoo Finance API (yfinance).
Technical Indicators: Visualize Moving Averages (100, 200, and 250 days) for trend analysis.
Stock Price Prediction: Predict stock prices for the next 10 or 20 days using a trained LSTM model.
Comparison of Trends: Display original vs predicted prices for better decision-making.
Interactive Visualizations: Use Matplotlib and Streamlit for dynamic and user-friendly charts.

### 🛠️ Tech Stack
Frontend: Streamlit (for building the user interface)
Backend: Python
Machine Learning: Keras, TensorFlow (LSTM model for time-series forecasting)
Data Retrieval: Yahoo Finance API (yfinance)
Visualization: Matplotlib, Pandas

### 📂 Project Structure
Stock_Price_Predictor/

│── stock_predictor.py            # Main Streamlit application

│── Stock market Youtube.ipynb    # Jupyter Notebook for ML analysis

│── Latest_stock_price_model.keras # Pretrained LSTM model

│── requirements.txt               # Python dependencies

│── README.md                      # Documentation

│── .gitignore                     # Ignore unnecessary files


### 🎯 How to Run Locally

### 1️⃣ Clone the Repository
git clone <Repository Link>

cd Stock-Price-Predictor

### 2️⃣ Create & Activate Virtual Environment (Recommended)
python -m venv venv

source venv/bin/activate  # On macOS/Linux

venv\Scripts\activate  # On Windows

### 3️⃣ Install Dependencies
pip install -r requirements.txt

### 4️⃣ Run the Application
streamlit run stock_predictor.py

The app will be available at http://localhost:8501/.

### 📝 Running the Jupyter Notebook
The Jupyter Notebook (Stock market Youtube.ipynb) provides an in-depth analysis of stock market trends and predictions. To run it:

Open the notebook locally:
jupyter notebook
Alternatively, upload it to Google Colab for cloud-based execution.



### 👨‍💻 Darshan Chavda
