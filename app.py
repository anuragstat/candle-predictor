
import streamlit as st
import pandas as pd
import joblib
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
from ta.trend import EMAIndicator, SMAIndicator

st.set_page_config(page_title="Candle Predictor", layout="centered")

st.title("📈 Candle Predictor – S&P 500 (Gradient Boosting Model)")
st.markdown("Upload 40 days of OHLC data to predict tomorrow's candle color.")

uploaded_file = st.file_uploader("Upload CSV File", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.dropna(inplace=True)

    if df.shape[0] < 35:
        st.error("Please upload at least 35 rows of OHLC data.")
    else:
        # Pivot + R/S levels (using previous day)
        df['Pivot'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
        df['R1'] = 2 * df['Pivot'] - df['Low']
        df['S1'] = 2 * df['Pivot'] - df['High']
        df['R2'] = df['Pivot'] + (df['High'] - df['Low'])
        df['S2'] = df['Pivot'] - (df['High'] - df['Low'])
        df['R3'] = df['R1'] + (df['High'] - df['Low'])
        df['S3'] = df['S1'] - (df['High'] - df['Low'])

        # Indicators
        df['RSI'] = RSIIndicator(df['Close'], window=14).rsi()
        df['ATR'] = AverageTrueRange(df['High'], df['Low'], df['Close'], window=14).average_true_range()
        df['EMA_5'] = EMAIndicator(df['Close'], window=5).ema_indicator()
        df['SMA_21'] = SMAIndicator(df['Close'], window=21).sma_indicator()

        # Feature engineering
        df['RSI_Change'] = df['RSI'].diff()
        df['ATR_Change'] = df['ATR'].diff()
        df['EMA_Slope'] = df['EMA_5'].diff()
        df['SMA_Slope'] = df['SMA_21'].diff()
        df['Pivot_Change'] = df['Pivot'].diff()
        df['Crossover'] = 0
        df.loc[(df['EMA_5'] > df['SMA_21']) & (df['EMA_5'].shift(1) <= df['SMA_21'].shift(1)), 'Crossover'] = 1
        df.loc[(df['EMA_5'] < df['SMA_21']) & (df['EMA_5'].shift(1) >= df['SMA_21'].shift(1)), 'Crossover'] = -1

        # Select the last row
        last_row = df.dropna().iloc[-1:]

        # Load model
        model = joblib.load("gradient_boosting_candle_color_predictor.pkl")

        # Features used in model
        features = [
            'Pivot', 'R1', 'S1', 'R2', 'S2', 'R3', 'S3',
            'RSI', 'ATR', 'EMA_5', 'SMA_21', 'Crossover',
            'RSI_Change', 'ATR_Change', 'EMA_Slope', 'SMA_Slope', 'Pivot_Change'
        ]

        X = last_row[features]
        prediction = model.predict(X)[0]
        confidence = model.predict_proba(X).max(axis=1)[0] * 100

        label = "🟢 Green (Up)" if prediction == 1 else "🔴 Red (Down)"
        st.success(f"**Predicted Candle Color: {label}**")
        st.info(f"**Confidence:** {confidence:.2f}%")

        with st.expander("🔍 Features Used"):
            st.dataframe(X.T)
