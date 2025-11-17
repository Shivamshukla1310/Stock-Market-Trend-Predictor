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
# Cache the loaded model so it isn't reloaded from disk on every interaction
# Using cache_resource instead of cache_data because the object is a heavy ML model, not plain data
@st.cache_resource
def load_my_model(path="stock_dl_model.h5"):
    return load_model(path) # Loads a pre-trained deep learning model (Keras .h5 format)

# Load the model once and reuse it - avoid huge inference delays and RAM spikes
model = load_my_model()

# ------------------------- MAIN LOGIC -------------------------
# Triggere data fetch and analysis only when user clicks (prevents automatic API calls and wasted compute)
if st.button("🚀 Fetch & Analyze"):

    # Shown a temporary loading indicator while pulling historical price data from Yahoo Finance
    with st.spinner("📥 Downloading stock data..."):
        df = yf.download(symbol, start=start_date, end=end_date)

    # Defensive check - API may return empty results for invalid tickers, holidays, or bad date ranges
    if df.empty:
        st.error("No data found for this ticker & date range.")
    else:
        st.success(f"✅ Data successfully fetched for **{symbol}** ({len(df)} days).")

        # Convert index to a useable column and enforce proper date type + ordering for plotting + modeling
        df.reset_index(inplace=True)
        df['Date'] = pd.to_datetime(df['Date'])
        df.sort_values('Date', inplace=True)

        # Calculate multiple Exponential Moving Averages (used trend indicators at different horizons
        for span in [20, 50, 100, 200]:
            df[f'EMA{span}'] = df['Close'].ewm(span=span, adjust=False).mean()

        # Separate UI sections for clarity - avoid cluttered single-page experience
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "💹 EMAs", "🤖 Predictions", "📈 Insights"])

        # ------------------------- TAB 1: Overview -------------------------
        with tab1:
            st.subheader("📉 Candlestick & Volume Chart")

           # Enforce proper datetime type; `errors='coerce'` avoids crashes if Yahoo returns malformed dates
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

            # # Optional safety step for bad data; kept commented in case user wants raw inspection
            # df.dropna(subset=['Date', 'Open', 'High', 'Low', 'Close'], inplace=True)

            # Initialize an empty Plotly figure so multiple traces can be layered cleanly
            fig = go.Figure()

            # Price movement via candlestick (standard for technical analysis)
            fig.add_trace(go.Candlestick(
                x=df['Date'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='Price'
            ))

            # Add volume bars on a secondary Y-axis (scaled to millions for readability)
            fig.add_trace(go.Bar(
                x=df['Date'],
                y=df['Volume'] / 1e6,
                name='Volume (M)',
                opacity=0.3, # makes sure candlesticks remian visually dominant
                yaxis='y2' # Assign to secondary axis instead of stacking on price values
            ))

            # Configure axes, theme, and chart layout for clarity and professional look
            fig.update_layout(
                xaxis_rangeslider_visible=False,
                yaxis_title="Price",
                yaxis2=dict(title="Volume (M)", overlaying="y", side="right"),
                template="plotly_dark",
                height=600,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )

            # Render chart once (important — calling twice causes duplicate visual glitches)
            st.plotly_chart(fig, use_container_width=True)

            # Show most recent rows for quick sanity check of data quality & indicator calculations
            st.dataframe(df.tail(), use_container_width=True)

        # ------------------------- TAB 2: EMA Trends -------------------------
        with tab2:
            st.subheader("📈 Exponential Moving Averages (Trend Visualization)")

            # Ensure consistent datetime format for plotting and potential tooltip usage
            df['Date'] = pd.to_datetime(df['Date'])
            df['Date_str'] = df['Date'].dt.strftime('%Y-%m-%d')

            # Initialize chart for EMAs + price overlay
            ema_fig = go.Figure()

            # Baseline reference curve: actual closing price
            ema_fig.add_trace(go.Scatter(
                x=df['Date'],
                y=df['Close'],
                name='Close',
                line=dict(color='gold', width=2) # Brighter + thicker to visually anchor the trend
            ))

            # Color mapping for EMA lines — chosen to differentiate short vs long-term trends
            ema_colors = {'EMA20': 'green', 'EMA50': 'red', 'EMA100': 'purple', 'EMA200': 'blue'}

            # Add each EMA line to the chart (loop avoids repetitive code)
            for col, color in ema_colors.items():
                ema_fig.add_trace(go.Scatter(
                    x=df['Date'],
                    y=df[col],
                    name=col,
                    line=dict(color=color)
                ))

            # Layout tweaks for readability + dark mode consistency
            ema_fig.update_layout(
                template="plotly_dark",
                height=500,
                xaxis_title="Date",
                yaxis_title="Price",
                legend_title="Indicators"
            )

            # Render final plot
            st.plotly_chart(ema_fig, use_container_width=True)

        # ------------------------- TAB 3: Predictions -------------------------
        with tab3:
            st.subheader("🤖 Model Predictions vs Actual Prices")

            # Use only the Close column for training/prediction (model assumes univariate time series)
            close = df[['Close']].reset_index(drop=True)
            
            # Hard-coded train/test split (70/30). Not time-shuffled → preserves chronological integrity.
            split = int(len(close) * 0.7)
            train = close[:split]
            test = close[split:]
            
            # Normalization required because the model was trained on MinMax-scaled values, not raw prices
            scaler = MinMaxScaler()
            train_scaled = scaler.fit_transform(train)

            # Construct prediction input by taking the last 100 training points + full test period
            # Using past context is critical; otherwise the model has no previous state.
            past_100 = train.tail(100).reset_index(drop=True)
            final_df = pd.concat([past_100, test.reset_index(drop=True)], ignore_index=True)
            
            # Scale full inference window using the same scaler → avoids distribution mismatch
            input_data = scaler.transform(final_df)

            # Sliding window creation to reproduce input shape expected by LSTM/CNN model (100 timesteps)
            x_test = []
            for i in range(100, input_data.shape[0]):
                x_test.append(input_data[i-100:i])
            x_test = np.array(x_test)

            # Generate predictions — model outputs scaled values so they must be inverted to real prices
            preds = model.predict(x_test)
            preds = scaler.inverse_transform(preds).flatten()
            
            # Ground truth values aligned to the same prediction horizon (exclude first 100 warm-up rows)
            actual = scaler.inverse_transform(input_data[100:].reshape(-1,1)).flatten()
            
            # Keep consistent date formatting for potential labeling or advanced hover display
            df['Date'] = pd.to_datetime(df['Date'])
            df['Date_str'] = df['Date'].dt.strftime('%Y-%m-%d')

            # ------------------------- Visualization: Actual vs Predicted -------------------------
            comp = go.Figure()
            comp.add_trace(go.Scatter(y=actual, mode='lines', name='Actual', line=dict(color='green')))
            comp.add_trace(go.Scatter(y=preds, mode='lines', name='Predicted', line=dict(color='red')))
            
            comp.update_layout(
                height=500,
                title="Actual vs Predicted Closing Prices",
                template="plotly_dark",
                xaxis_title="Days", # Index-based, NOT calendar-based (important limitation!)
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
