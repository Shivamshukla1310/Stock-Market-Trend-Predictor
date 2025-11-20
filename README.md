<h1 align="center">Stock Market Trend Predictor</h1>
<p align="center">
  Time-Series Analysis • Deep Learning • Interactive Financial Visualization
</p>

---

### 📘 Overview

This project is an end-to-end **Stock Market Trend Predictor** built using Streamlit, TensorFlow, and Yahoo Finance data.  
It analyzes historical stock prices, computes technical indicators, and predicts future stock trends using a trained LSTM model.

The application provides a clean, interactive dashboard where users can view:

- Historical candlestick patterns  
- Volume analysis  
- EMA trends (20, 50, 100, 200)  
- Predicted vs Actual price comparison  
- Next-day stock trend insights  

This project demonstrates real-world use of **ETL, Time-Series Forecasting, Feature Engineering, Deep Learning, and Data Visualization**.

---

### 🔗 Live Demo

https://stock-market-trend-predictor-trbsdvd5qtku3msgn9uqqo.streamlit.app/

---

### 🖼️ Screenshots

*(Add your images inside the repository and replace these paths accordingly)*

<p align="center">
  <img src="https://github.com/Shivamshukla1310/Nifty-Radar/blob/main/Screenshot%202025-11-16%20160629.png" width="70%" />
</p>

<p align="center">
  <img src=""C:\Users\Hp\OneDrive\Pictures\Screenshots\Screenshot 2025-11-16 160629.png"" width="40%" />
</p>

---

### 🔍 Features

- Fetches **live & historical market data**  
- Visualizes candlestick charts using Plotly  
- Computes technical indicators (EMA20/50/100/200)  
- LSTM Deep Learning model for **next-day price prediction**  
- Trend classification: *Uptrend / Downtrend / Neutral*  
- Downloadable processed dataset  
- Clean tab-based UI built with Streamlit  

---

### 🧠 Machine Learning Scope

This project uses Deep Learning for **time-series forecasting**.

#### Covered ML Concepts:
- Data scaling using MinMaxScaler  
- Converting time series into supervised learning using **sliding windows (100 timesteps)**  
- Training an LSTM model for sequence prediction  
- Inference pipeline with inverse transformation  
- Comparing actual vs predicted closing prices  

#### ETL Pipeline:
1. **Extract** — Data pulled from Yahoo Finance  
2. **Transform** — Cleaning, EMA feature engineering, scaling  
3. **Load / Predict** — Feeding sequences to LSTM model + visualization  

---

### 🧱 Tech Stack

**Frontend / UI**
- Streamlit
- Plotly

**Backend / Logic**
- Python  
- yfinance  
- pandas  
- numpy  
- scikit-learn  

**Machine Learning**
- TensorFlow / Keras (LSTM Model)
- MinMaxScaler preprocessing  
- Time-Series forecasting pipeline  

**Deployment**
- Streamlit Cloud

---

### 📂 Project Structure
Stock-Market-Trend-Predictor/
│
├── streamlit_app.py # Main Streamlit application
├── stock_dl_model.h5 # Trained LSTM model
├── requirements.txt # Dependencies
├── assets/ # Screenshots and media
└── README.md # Documentation


---

### ⚙️ Installation

```bash
git clone https://github.com/YOUR_USERNAME/Stock-Market-Trend-Predictor.git
cd Stock-Market-Trend-Predictor
pip install -r requirements.txt
streamlit run streamlit_app.py

📈 How Predictions Work

Load historical closing prices

Apply MinMax scaling

Create 100-step sliding windows

Feed sequences into LSTM

Generate next-day prediction

Inverse transform values

Classify trend

🚀 Future Enhancements

Multi-feature LSTM (OHLC + Volume + Indicators)

Sentiment analysis from financial news

Pattern detection using CNN

Live intraday predictions

Backtesting module for strategy evaluation

Multi-stock comparison dashboard

👨‍💻 Author

Shivam Kumar Shukla
Python Developer • ML Enthusiast • Full-Stack Learner

⭐ Support

If this project helps you, consider giving it a star on GitHub!
