# import streamlit as st
# import yfinance as yf
# import pandas as pd
# import numpy as np
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout
# from tensorflow.keras.callbacks import EarlyStopping
# import datetime

# st.set_page_config(page_title="Stock Trend Predictor", layout="wide")

# # -------------------- Helper functions --------------------
# @st.cache_data(show_spinner=False)
# def fetch_data(ticker: str, start: str, end: str) -> pd.DataFrame:
#     """Fetches historical OHLCV data from Yahoo Finance."""
#     df = yf.download(ticker, start=start, end=end, progress=False)
#     if df.empty:
#         raise ValueError("No data fetched. Check ticker symbol or date range.")
#     df.reset_index(inplace=True)
#     df.rename(columns={"Date": "Date"}, inplace=True)
#     return df


# def compute_ema(df: pd.DataFrame, span: int = 20) -> pd.Series:
#     return df['Close'].ewm(span=span, adjust=False).mean()


# def create_lstm_dataset(series: np.ndarray, look_back: int = 60):
#     x, y = [], []
#     for i in range(look_back, len(series)):
#         x.append(series[i - look_back:i, 0])
#         y.append(series[i, 0])
#     x, y = np.array(x), np.array(y)
#     x = np.reshape(x, (x.shape[0], x.shape[1], 1))
#     return x, y


# # -------------------- Sidebar: Inputs --------------------
# st.sidebar.title("Settings")
# with st.sidebar.form(key='inputs'):
#     ticker = st.text_input('Stock ticker (e.g. AAPL, MSFT, TCS.NS)', value='AAPL')
#     col1, col2 = st.columns(2)
#     with col1:
#         start_date = st.date_input('Start date', value=datetime.date.today() - datetime.timedelta(days=365*2))
#     with col2:
#         end_date = st.date_input('End date', value=datetime.date.today())
#     ema_span = st.number_input('EMA span (days)', min_value=5, max_value=200, value=20)
#     look_back = st.slider('LSTM look-back (days)', min_value=10, max_value=180, value=60)
#     test_size = st.slider('Test size (%)', min_value=5, max_value=50, value=20)
#     epochs = st.slider('LSTM epochs', min_value=1, max_value=50, value=10)
#     batch_size = st.selectbox('Batch size', options=[8, 16, 32, 64], index=2)
#     submit = st.form_submit_button('Run Analysis')


# if not submit:
#     st.title("Stock Trend Predictor — Enter inputs and click 'Run Analysis'")
#     st.info("This app fetches data from Yahoo Finance, shows analysis charts, computes EMA, plots candlesticks, trains an LSTM, and displays predicted vs actual prices.")
#     st.stop()


# # -------------------- Data fetch and basic checks --------------------
# st.title(f"Stock Trend Predictor — {ticker.upper()}")
# try:
#     with st.spinner('Fetching data...'):
#         df = fetch_data(ticker.upper(), start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
# except Exception as e:
#     st.error(f"Error fetching data: {e}")
#     st.stop()

# st.markdown(f"**Data range:** {df['Date'].min().date()} — {df['Date'].max().date()}  **Rows:** {len(df)}")

# # -------------------- Overview & Data Table --------------------
# st.subheader('OHLCV Data (sample)')
# st.dataframe(df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].tail(50))

# # -------------------- Charts: High/Low/Open/Close/Volume --------------------
# st.subheader('Price & Volume Charts')
# price_fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25])
# price_fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Close', mode='lines'), row=1, col=1)
# price_fig.add_trace(go.Scatter(x=df['Date'], y=df['Open'], name='Open', mode='lines'), row=1, col=1)
# price_fig.add_trace(go.Scatter(x=df['Date'], y=df['High'], name='High', mode='lines'), row=1, col=1)
# price_fig.add_trace(go.Scatter(x=df['Date'], y=df['Low'], name='Low', mode='lines'), row=1, col=1)
# price_fig.add_trace(go.Bar(x=df['Date'], y=df['Volume'], name='Volume', showlegend=False), row=2, col=1)
# price_fig.update_layout(height=600, legend=dict(orientation='h'))
# st.plotly_chart(price_fig, use_container_width=True)

# # -------------------- EMA Calculation & Chart --------------------
# st.subheader(f'Exponential Moving Average (EMA, span={ema_span})')
# df['EMA'] = compute_ema(df, span=ema_span)
# ema_fig = go.Figure()
# ema_fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Close'))
# ema_fig.add_trace(go.Scatter(x=df['Date'], y=df['EMA'], name=f'EMA-{ema_span}'))
# ema_fig.update_layout(height=400)
# st.plotly_chart(ema_fig, use_container_width=True)

# # -------------------- Candlestick Chart --------------------
# st.subheader('Candlestick Chart')
# candle = go.Figure(data=[go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Candlestick')])
# # Overlay EMA if available
# candle.add_trace(go.Scatter(x=df['Date'], y=df['EMA'], name=f'EMA-{ema_span}', line=dict(width=1)))
# candle.update_layout(height=600)
# st.plotly_chart(candle, use_container_width=True)

# # -------------------- LSTM Prediction --------------------
# st.subheader('LSTM Time-Series Forecast (Close Price)')
# # Prepare dataset using only 'Close' price
# data_close = df[['Date', 'Close']].copy()
# scaler = MinMaxScaler(feature_range=(0, 1))
# scaled_close = scaler.fit_transform(data_close[['Close']].values)

# # split
# split_idx = int(len(scaled_close) * (100 - test_size) / 100)
# train_data = scaled_close[:split_idx]
# test_data = scaled_close[split_idx - look_back:]

# # create datasets
# x_train, y_train = create_lstm_dataset(train_data, look_back=look_back)
# x_test, y_test = create_lstm_dataset(test_data, look_back=look_back)

# # build model
# model = Sequential()
# model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
# model.add(Dropout(0.2))
# model.add(LSTM(50))
# model.add(Dropout(0.2))
# model.add(Dense(1))
# model.compile(optimizer='adam', loss='mean_squared_error')

# st.text('Model summary:')
# model_summary = []
# model.summary(print_fn=lambda x: model_summary.append(x))
# st.code('\n'.join(model_summary))

# with st.spinner('Training LSTM model (this may take a while)...'):
#     es = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
#     history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[es])

# # predictions
# predicted_train = model.predict(x_train)
# predicted_test = model.predict(x_test)

# # inverse transform
# predicted_train = scaler.inverse_transform(predicted_train)
# predicted_test = scaler.inverse_transform(predicted_test.reshape(-1, 1))
# actual_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# # Create a results DataFrame aligned with dates
# test_dates = data_close['Date'].iloc[split_idx:split_idx + len(actual_test)].reset_index(drop=True)
# results_df = pd.DataFrame({'Date': test_dates, 'Actual': actual_test.ravel(), 'Predicted': predicted_test.ravel()})

# st.markdown('**Prediction results (test set)**')
# st.dataframe(results_df.tail(50))

# # Plot actual vs predicted
# fig_pred = go.Figure()
# fig_pred.add_trace(go.Scatter(x=results_df['Date'], y=results_df['Actual'], name='Actual'))
# fig_pred.add_trace(go.Scatter(x=results_df['Date'], y=results_df['Predicted'], name='Predicted'))
# fig_pred.update_layout(height=500)
# st.plotly_chart(fig_pred, use_container_width=True)

# # Percentage change and next-day prediction
# last_actual = results_df['Actual'].iloc[-1]
# last_pred = results_df['Predicted'].iloc[-1]
# perc_change = ((last_pred - last_actual) / last_actual) * 100

# st.metric(label='Last actual vs last predicted (test set)', value=f"{last_pred:.2f}", delta=f"{perc_change:.2f}%")

# # Simple insight engine
# st.subheader('Investment Insight (simple rule-based)')
# insight = ''
# if perc_change > 1.0:
#     insight = f"Predicted to rise by {perc_change:.2f}% — Consider BUY or further fundamental analysis."
# elif perc_change < -1.0:
#     insight = f"Predicted to fall by {abs(perc_change):.2f}% — Consider SELL or set stop-loss."
# else:
#     insight = f"Predicted change small ({perc_change:.2f}%) — Consider HOLD and monitor." 

# st.info(insight)

# st.markdown('---')
# st.caption('Notes: This app demonstrates a simple LSTM-based approach for educational purposes. Do not treat this as investment advice. Model performance depends heavily on hyperparameters, data quality, and market conditions.')

# # -------------------- Allow user to download the results as CSV --------------------
# @st.cache_data

# def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
#     return df.to_csv(index=False).encode('utf-8')

# csv_bytes = df_to_csv_bytes(results_df)
# st.download_button('Download predictions CSV', data=csv_bytes, file_name=f'{ticker}_predictions.csv', mime='text/csv')

# # -------------------- End --------------------



# # import streamlit as st
# # import plotly.graph_objects as go
# # import pandas as pd
# # from datetime import datetime

# # df = pd.DataFrame({
# #     "Date": pd.date_range("2024-01-01", periods=10),
# #     "Open": range(10),
# #     "High": [x+2 for x in range(10)],
# #     "Low": [x-1 for x in range(10)],
# #     "Close": [x+1 for x in range(10)]
# # })

# # fig = go.Figure(data=[go.Candlestick(
# #     x=df['Date'].dt.to_pydatetime(),
# #     open=df['Open'],
# #     high=df['High'],
# #     low=df['Low'],
# #     close=df['Close']
# # )])
# # st.plotly_chart(fig, use_container_width=True)
