# Core libraries for UI, data handling, math, API, charts, and ML model
import streamlit as st # for building interactive web apps
import pandas as pd # for data manipulation and analysis
import numpy as np # for numerical operations
import yfinance as yf # to fetch historical stock / financial data
from datetime import datetime # for handling and formating dates
import plotly.graph_objects as go # for building interactive visual charts 
from tensorflow.keras.models import load_model # to load pre-trained ML / Deep learning models
from sklearn.preprocessing import MinMaxScaler # for scalling data into 0-1 range before model prediction

# ------------------------- PAGE SETUP -------------------------
st.set_page_config(
    layout="wide", # expands content to full browser width instead of narrow default
    page_title="Stock Trend Predictor" # title shown in browser tab, useful for clarity & branding
)
# Display a main heading at the top of the app UI
st.title("📈 Stock Trend Predictor (Smart Prototype)") # immediate user-facing title for cantent

# ------------------------- SIDEBAR -------------------------
# sider block for user inputs control (keep UI clean and separated from main view)
with st.sidebar:
    st.header("🔍 Stock Settings")
    
    # user-entered stock ticker symbol (default set up to a reliable Indian stock to avoid empty state)
    symbol = st.text_input("Stock ticker (e.g. POWERGRID.NS, AAPL)", "POWERGRID.NS")
    
    # Historical date start date (default chosen far back so long-term training / testing is possible)
    start_date = st.date_input("Start date", value=pd.to_datetime("2000-01-01"))
    
    # end date automatically bound to today's date to ensure no invalid range
    end_date = st.date_input("End date", value=datetime.now().date())
    
    # short guidance to reduce user mistakes with exchange-specific ticker formats
    st.caption("Tip: Use NSE tickers with .NS or .BO suffix")

# ------------------------- LOAD MODEL -------------------------
@st.cache_resource
def load_my_model(path="stock_dl_model.h5"):
    return load_model(path)

model = load_my_model()

# ------------------------- MAIN LOGIC -------------------------
if st.button("🚀 Fetch & Analyze"):
    with st.spinner("📥 Downloading stock data..."):
        df = yf.download(symbol, start=start_date, end=end_date)

    if df.empty:
        st.error("No data found for this ticker & date range.")
    else:
        st.success(f"✅ Data successfully fetched for **{symbol}** ({len(df)} days).")

        # Ensure datetime index
        df.reset_index(inplace=True)
        df['Date'] = pd.to_datetime(df['Date'])
        df.sort_values('Date', inplace=True)

        # Compute EMAs
        for span in [20, 50, 100, 200]:
            df[f'EMA{span}'] = df['Close'].ewm(span=span, adjust=False).mean()

        # Layout tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "💹 EMAs", "🤖 Predictions", "📈 Insights"])

        # ------------------------- TAB 1: Overview -------------------------
        with tab1:
            st.subheader("📉 Candlestick & Volume Chart")

           # Ensure Date is datetime
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

            # # Drop missing essential columns
            # df.dropna(subset=['Date', 'Open', 'High', 'Low', 'Close'], inplace=True)

            # Create figure
            fig = go.Figure()

            # Candlestick
            fig.add_trace(go.Candlestick(
                x=df['Date'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='Price'
            ))

            # Volume overlay
            fig.add_trace(go.Bar(
                x=df['Date'],
                y=df['Volume'] / 1e6,
                name='Volume (M)',
                opacity=0.3,
                yaxis='y2'
            ))

            # Layout
            fig.update_layout(
                xaxis_rangeslider_visible=False,
                yaxis_title="Price",
                yaxis2=dict(title="Volume (M)", overlaying="y", side="right"),
                template="plotly_dark",
                height=600,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )

            # ✅ Only one plot call
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df.tail(), use_container_width=True)

        # ------------------------- TAB 2: EMA Trends -------------------------
        with tab2:
            st.subheader("📈 Exponential Moving Averages (Trend Visualization)")
            df['Date'] = pd.to_datetime(df['Date'])
            df['Date_str'] = df['Date'].dt.strftime('%Y-%m-%d')

            ema_fig = go.Figure()
            ema_fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Close', line=dict(color='gold', width=2)))
            ema_colors = {'EMA20': 'green', 'EMA50': 'red', 'EMA100': 'purple', 'EMA200': 'blue'}
            for col, color in ema_colors.items():
                ema_fig.add_trace(go.Scatter(x=df['Date'], y=df[col], name=col, line=dict(color=color)))

            ema_fig.update_layout(
                template="plotly_dark",
                height=500,
                xaxis_title="Date",
                yaxis_title="Price",
                legend_title="Indicators"
            )
            st.plotly_chart(ema_fig, use_container_width=True)

        # ------------------------- TAB 3: Predictions -------------------------
        with tab3:
            st.subheader("🤖 Model Predictions vs Actual Prices")

            close = df[['Close']].reset_index(drop=True)
            split = int(len(close) * 0.7)
            train = close[:split]
            test = close[split:]
            scaler = MinMaxScaler()
            train_scaled = scaler.fit_transform(train)

            # Prepare test set
            past_100 = train.tail(100).reset_index(drop=True)
            final_df = pd.concat([past_100, test.reset_index(drop=True)], ignore_index=True)
            input_data = scaler.transform(final_df)

            x_test = []
            for i in range(100, input_data.shape[0]):
                x_test.append(input_data[i-100:i])
            x_test = np.array(x_test)

            # Predict
            preds = model.predict(x_test)
            preds = scaler.inverse_transform(preds).flatten()
            actual = scaler.inverse_transform(input_data[100:].reshape(-1,1)).flatten()
            df['Date'] = pd.to_datetime(df['Date'])
            df['Date_str'] = df['Date'].dt.strftime('%Y-%m-%d')

            # Chart
            comp = go.Figure()
            comp.add_trace(go.Scatter(y=actual, mode='lines', name='Actual', line=dict(color='green')))
            comp.add_trace(go.Scatter(y=preds, mode='lines', name='Predicted', line=dict(color='red')))
            comp.update_layout(
                height=500,
                title="Actual vs Predicted Closing Prices",
                template="plotly_dark",
                xaxis_title="Days",
                yaxis_title="Price"
            )
            st.plotly_chart(comp, use_container_width=True)

            # Metrics
            last_pred = preds[-1]
            last_actual = actual[-1]
            change = ((last_pred - last_actual) / last_actual) * 100
            st.metric(label="Predicted Change (%)", value=f"{change:.2f}%")

        # ------------------------- TAB 4: Insights -------------------------
        with tab4:
            st.subheader("📊 Investment Insight")
            if change > 2:
                st.success("📈 The model indicates a potential **Uptrend**. Positive momentum detected — may be a good time to consider short-term investment.")
            elif change < -2:
                st.error("📉 The model suggests a **Downtrend**. Prices might fall — consider caution or waiting for a reversal.")
            else:
                st.warning("⚖️ The trend appears **Neutral**. Wait for stronger signals before making a move.")

            st.info(f"**Last predicted closing price:** {last_pred:.2f}\n\n**Last actual closing price:** {last_actual:.2f}")

            st.caption("Upcoming: AI-based Sentiment Analysis, Risk Scoring, and News Feed Integration 📡")

        # ------------------------- CSV DOWNLOAD -------------------------
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="💾 Download dataset (CSV)",
            data=csv,
            file_name=f"{symbol}_data.csv",
            mime='text/csv'
        )
