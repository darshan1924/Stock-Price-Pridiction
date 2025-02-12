# 📈 Stock Price Predictor App

## 🚀 Overview
The **Stock Price Predictor App** is a machine learning-powered web application built with **Streamlit**. It allows users to:
- Fetch real-time stock data using **Yahoo Finance** (`yfinance`).
- Visualize **Moving Averages** (100, 200, and 250 days).
- Predict stock prices for the next **10 or 20 days** using a trained deep learning model (`.keras`).
- Display **original vs predicted prices** for better trend analysis.

---

## 📌 Features
✔️ Real-time stock data fetching 📊  
✔️ Interactive visualizations (Matplotlib & Streamlit) 📈  
✔️ Machine Learning predictions using **Keras LSTM model** 🤖  
✔️ Supports all **publicly traded stocks** via Yahoo Finance 🏰  

---

## 🎥 Live Demo
[![Streamlit App](https://img.shields.io/badge/Live-Demo-green?style=for-the-badge&logo=streamlit)](https://your-app-name.streamlit.app)

---

## 🛠️ Tech Stack
- **Frontend**: Streamlit
- **Backend**: Python
- **Machine Learning**: Keras, TensorFlow
- **Data Retrieval**: Yahoo Finance API (`yfinance`)
- **Visualization**: Matplotlib, Pandas

---

## 📂 Project Structure
```
Stock_Price_Predictor/
│── stock_predictor.py            # Streamlit App
│── Stock market Youtube.ipynb    # Jupyter Notebook for ML Analysis
│── Latest_stock_price_model.keras # Pretrained ML model
│── requirements.txt               # Dependencies
│── README.md                      # Documentation
│── .gitignore                      # Ignore unnecessary files
```

---

## 🎯 How to Run Locally

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/Stock-Price-Predictor.git
cd Stock-Price-Predictor
```

### 2️⃣ Create & Activate Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate  # On Windows
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Run Streamlit App
```bash
streamlit run stock_predictor.py
```
🚀 Your app will be available at `http://localhost:8501/`  

---

## 🌎 Deploying on Streamlit Cloud
1️⃣ Push this project to **GitHub**  
2️⃣ Go to [Streamlit Cloud](https://share.streamlit.io/)  
3️⃣ Select your repository & deploy 🎉  

---

## 📝 Running Jupyter Notebook (`Stock market Youtube.ipynb`)
To analyze stock market trends and predictions:
1. Open the **Jupyter Notebook** locally:
   ```bash
   jupyter notebook
   ```
2. OR Open it directly in **Google Colab**:  
   [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/Stock-Price-Predictor/blob/main/Stock%20market%20Youtube.ipynb)

---

## ⚡ Future Improvements
- ✅ Add sentiment analysis for stock market news 📰  
- ✅ Implement **Deep Learning Transformers** (LSTM, GRU) 🔥  
- ✅ Deploy as a **fully functional web app**  

---

## 📌 Contributing
Contributions are welcome!  
Feel free to **fork** this repo and submit a **pull request**.  

---

## 🛡️ License
This project is licensed under the **MIT License**.  

---

## 💡 Author
**👨‍💻 Darshan Chavda**  
🔗 [GitHub](https://github.com/yourusername)  
🔗 [LinkedIn](https://linkedin.com/in/yourusername)  

import logging

# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a file handler and a stream handler
file_handler = logging.FileHandler('stock_price_predictor.log')
stream_handler = logging.StreamHandler()

# Create a formatter and set it for the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# Example usage:
logger.info('Stock Price Predictor App started')
logger.debug('Debug message')
logger.warning('Warning message')
logger.error('Error message')
logger.critical('Critical message')