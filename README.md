🌟 Stock Market Trend Predictor — Streamlit Web Application

An intelligent, interactive web application that predicts stock market trends using Deep Learning (LSTM), technical indicators, and live market data from Yahoo Finance.
Built using Streamlit, Plotly, TensorFlow, and yfinance, this project helps users analyze market patterns and visualize candlesticks, EMAs, volumes, and ML-based predictions.

🔗 Live Demo

👉 (https://stock-market-trend-predictor-trbsdvd5qtku3msgn9uqqo.streamlit.app/)


🖼️ Screenshots

Add your website images here (PC & Mobile view):

<img width="1360" height="623" alt="Screenshot 2025-11-16 223939" src="https://github.com/user-attachments/assets/4e10b753-b1a7-4b2e-bcc1-d36cfa675d28" />
<img width="1362" height="623" alt="Screenshot 2025-11-16 160629" src="https://github.com/user-attachments/assets/f13e1fc2-4a1e-4c9d-9e21-dafccc468673" />
<img width="958" height="452" alt="Screenshot 2025-11-16 160351" src="https://github.com/user-attachments/assets/99d4f47b-2170-44a2-b3d1-9adaac175bdf" />
![WhatsApp Image 2025-11-16 at 16 08 15_0140b38c](https://github.com/user-attachments/assets/9c962eb5-f3b4-4c86-82b3-52f29e5dab3e)
![WhatsApp Image 2025-11-16 at 16 08 19_4f7b6a36](https://github.com/user-attachments/assets/22ff9d39-0134-4797-9cc7-105f4b7bfee4)






	
📊 Charts

You can add your candlestick, EMA, volume, prediction charts here:

Candlestick Chart



Volume Chart

EMA Trend Chart

Prediction vs Actual Chart

🚀 Features
📥 Real-Time Stock Data Fetching

Pulls live & historical data using Yahoo Finance API

Supports Indian (NSE/BSE) and global tickers like:

POWERGRID.NS, INFY.NS, AAPL, TSLA

📊 Interactive Financial Charts

Candlestick chart

Volume bars (dual-axis)

EMA20, EMA50, EMA100, EMA200 trends

Zoom, hover, drag enabled via Plotly

🤖 ML-Based Stock Prediction

LSTM model predicts next-day price trend

Compares actual vs predicted

Shows % predicted change

Trend insight: Uptrend / Downtrend / Neutral

💾 Dataset Export

Download processed dataset (features + indicators) as CSV

🧠 Optimized App Experience

Cached ML model loading

Clean sidebar inputs

Tab-based sections for better UX

🧱 Project Structure
📦 Stock-Market-Trend-Predictor
│
├── streamlit_app.py          # Main Streamlit app
├── stock_dl_model.h5         # Trained deep learning model
├── requirements.txt          # Dependencies
├── README.md                 # Documentation
└── ...

🛠️ Tech Stack
Frontend / UI

Streamlit

Plotly

Backend / Logic

Python

yfinance

pandas, numpy

scikit-learn

TensorFlow / Keras

ML Model

LSTM (Long Short-Term Memory neural network)

Trained on historical closing prices

MinMax scaling

Sliding window (100-day sequence)

⚙️ Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/YOUR_USERNAME/Stock-Market-Trend-Predictor.git
cd Stock-Market-Trend-Predictor

2️⃣ Install Requirements
pip install -r requirements.txt

3️⃣ Run the App
streamlit run streamlit_app.py

📌 How Predictions Work
✔ Step 1: Load historical closing prices
✔ Step 2: Scale using MinMaxScaler
✔ Step 3: Create 100-day sliding windows
✔ Step 4: Feed into LSTM model
✔ Step 5: Inverse-transform predictions
✔ Step 6: Compare Actual vs Predicted
✔ Step 7: Show trend insight
📂 Features Explanation (Mapped to Tabs)
📊 Overview Tab

Candlestick chart

Volume chart

Recent data table

💹 EMA Tab

EMA20

EMA50

EMA100

EMA200

🤖 Predictions Tab

Actual vs Predicted chart

Trend %

LSTM inference

📈 Insights Tab

Uptrend / Downtrend / Neutral highlight

Future planned features:

Sentiment analysis

News & risk scoring

📄 Downloadable Dataset

Users can export the final processed dataset including:

Date

OHLC values

EMAs

Volume

Prediction data

🧪 Future Enhancements

Sentiment analysis using NLP

News-based trend scoring

Multi-stock comparison

Multi-feature ML models

Live intraday updates

Alerts & notifications

🏁 Conclusion

This project combines finance, machine learning, and interactive UI design to create a smart stock trend analysis tool. It serves as:

A practical learning project

A portfolio-ready ML application

A strong demonstration of full-stack Python + ML skills

👨‍💻 Author

Shivam Kumar Shukla
Python Developer • ML Enthusiast • Full-Stack Learner

⭐ Support the Project

If you like this project, don’t forget to ⭐️ star the repo!
