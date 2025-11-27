# app.py
import os
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, GRU, Dense, Input, MultiHeadAttention, LayerNormalization, Dropout, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from transformers import pipeline
from datetime import datetime
import math

# ---------------------------
# Ensure folders exist
# ---------------------------
os.makedirs("models", exist_ok=True)
os.makedirs("images", exist_ok=True)

# ---------------------------------
# Transformer model (Functional API)
# ---------------------------------
def build_transformer(input_shape):
    """
    input_shape: (timesteps, features)
    Returns a compiled Keras Model that outputs a single value (scaled).
    """
    inputs = Input(shape=input_shape)  # (timesteps, features)

    # MultiHeadAttention expects queries and values; using inputs for both.
    attn_output = MultiHeadAttention(num_heads=4, key_dim=16)(inputs, inputs)
    # Residual connection + layer norm
    x = LayerNormalization()(attn_output + inputs)

    # Feed-forward per time-step
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(32, activation="relu")(x)

    # Pool across time to get single vector
    x = GlobalAveragePooling1D()(x)
    outputs = Dense(1)(x)  # predicting scaled Close

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(0.001), loss="mse")
    return model

# ---------------------------------
# Streamlit UI
# ---------------------------------
st.set_page_config(page_title="📈 Stock Prediction Pro", layout="wide")
st.title("🚀 Stock Market Forecasting (TensorFlow Deep Learning)")
st.markdown("### Compare LSTM, GRU, and Transformer models with Sentiment Analysis")

# Sidebar configuration
st.sidebar.header("Configuration")
stock_symbol = st.sidebar.text_input("Stock Symbol (e.g., AAPL, INFY.NS):", "AAPL")
start_date = st.sidebar.date_input("Start Date", datetime(2018, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.today())
compare_all = st.sidebar.checkbox("Compare All Models", value=True)
n_days = st.sidebar.slider("Days to Forecast", 1, 7, 3)
train_epochs = st.sidebar.slider("Training Epochs", 5, 30, 10)

# Run button
if st.sidebar.button("Run Prediction"):
    with st.spinner("Fetching data and training models..."):
        # -----------------------
        # 1) Load Data
        # -----------------------
        df = yf.download(stock_symbol, start=start_date, end=end_date)

        if df.empty:
            st.error("No data fetched. Please check the stock symbol and date range.")
        else:
            df = df[["Open", "High", "Low", "Close", "Volume"]]
            st.subheader(f"📊 {stock_symbol} Price Data")
            st.line_chart(df["Close"])

            # -----------------------
            # 2) Preprocessing
            # -----------------------
            data = df.values.astype(float)  # shape: (samples, 5)
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data)

            def create_dataset(dataset, look_back=60):
                X, Y = [], []
                for i in range(look_back, len(dataset)):
                    X.append(dataset[i - look_back:i, :])  # shape: (look_back, features)
                    Y.append(dataset[i, 3])  # Close column (scaled)
                return np.array(X), np.array(Y)

            look_back = 60
            if len(scaled_data) <= look_back + 1:
                st.error(f"Not enough data for look_back={look_back}. Try an earlier start date.")
            else:
                X, Y = create_dataset(scaled_data, look_back)
                train_size = int(len(X) * 0.8)
                X_train, X_test = X[:train_size], X[train_size:]
                Y_train, Y_test = Y[:train_size], Y[train_size:]

                # -----------------------
                # 3) Define Models
                # -----------------------
                def build_lstm(input_shape):
                    m = Sequential([
                        LSTM(64, return_sequences=True, input_shape=input_shape),
                        LSTM(32),
                        Dense(1)
                    ])
                    m.compile(optimizer="adam", loss="mse")
                    return m

                def build_gru(input_shape):
                    m = Sequential([
                        GRU(64, return_sequences=True, input_shape=input_shape),
                        GRU(32),
                        Dense(1)
                    ])
                    m.compile(optimizer="adam", loss="mse")
                    return m

                input_shape = (look_back, X.shape[2])
                models = {
                    "LSTM": build_lstm(input_shape),
                    "GRU": build_gru(input_shape),
                    "Transformer": build_transformer(input_shape)
                }
                results = {}

                # -----------------------
                # 4) Train & Evaluate
                # -----------------------
                best_model_name = None
                if compare_all:
                    progress_bar = st.progress(0)
                    total = len(models)
                    idx = 0
                    for name, model in models.items():
                        idx += 1
                        st.info(f"Training {name} ...")
                        model.fit(X_train, Y_train, epochs=train_epochs, batch_size=32, verbose=0)
                        # Save model
                        model_path = f"models/{name.lower()}_model.h5"
                        try:
                            model.save(model_path)
                            st.text(f"Saved {name} to {model_path}")
                        except Exception as e:
                            st.warning(f"Could not save {name}: {e}")

                        # Predict on test
                        pred = model.predict(X_test)
                        # pred and Y_test are scaled close values; need to inverse-transform
                        inv_pred = scaler.inverse_transform(
                            np.concatenate((np.zeros((pred.shape[0], data.shape[1] - 1)), pred), axis=1)
                        )[:, -1]
                        inv_real = scaler.inverse_transform(
                            np.concatenate((np.zeros((Y_test.shape[0], data.shape[1] - 1)), Y_test.reshape(-1, 1)), axis=1)
                        )[:, -1]
                        rmse = math.sqrt(mean_squared_error(inv_real, inv_pred))
                        mae = mean_absolute_error(inv_real, inv_pred)
                        results[name] = {"pred": inv_pred, "real": inv_real, "RMSE": rmse, "MAE": mae}
                        progress_bar.progress(int((idx / total) * 100))

                    # Performance table
                    st.subheader("📊 Model Performance Comparison")
                    perf_df = pd.DataFrame({m: [r["RMSE"], r["MAE"]] for m, r in results.items()},
                                           index=["RMSE", "MAE"]).T
                    st.table(perf_df.round(3))

                    # Plot comparison
                    st.subheader("📉 Predictions Comparison (on test set)")
                    fig = go.Figure()
                    for name, r in results.items():
                        fig.add_trace(go.Scatter(y=r["pred"], mode="lines", name=f"{name} Pred"))
                    # show actual once (they all have same real)
                    any_real = next(iter(results.values()))["real"]
                    fig.add_trace(go.Scatter(y=any_real, mode="lines", name="Actual", line=dict(color="black", width=2)))
                    st.plotly_chart(fig, use_container_width=True)

                    # Choose best model (lowest RMSE)
                    best_model_name = min(results, key=lambda x: results[x]["RMSE"])
                    model = models[best_model_name]
                    st.success(f"🏆 Best Model: {best_model_name} (RMSE={results[best_model_name]['RMSE']:.3f})")

                else:
                    model_choice = st.sidebar.selectbox("Select Model", ["LSTM", "GRU", "Transformer"])
                    st.info(f"Training {model_choice} ...")
                    model = models[model_choice]
                    model.fit(X_train, Y_train, epochs=train_epochs, batch_size=32, verbose=0)
                    # Save the single model
                    model_path = f"models/{model_choice.lower()}_model.h5"
                    try:
                        model.save(model_path)
                        st.success(f"✔ {model_choice} model saved as {model_path}")
                    except Exception as e:
                        st.warning(f"Could not save {model_choice}: {e}")
                    best_model_name = model_choice

                # -----------------------
                # 5) Multi-Day Forecast (using chosen model)
                # -----------------------
                st.subheader(f"🔮 {n_days}-Day Forecast Ahead (using {best_model_name})")
                last_seq = scaled_data[-look_back:]  # shape: (look_back, features)
                forecast_scaled = []
                curr_seq = last_seq.copy()

                for _ in range(n_days):
                    pred_scaled = model.predict(curr_seq.reshape(1, look_back, X.shape[2]))  # (1,1)
                    # Build next input row in scaled space: keep zeros for other features (they'll be inverse-transformed later)
                    next_scaled = np.zeros((1, data.shape[1]))
                    next_scaled[0, 3] = pred_scaled[0, 0]  # scaled Close
                    forecast_scaled.append(pred_scaled[0, 0])
                    # Append to curr_seq and drop first row
                    curr_seq = np.vstack([curr_seq[1:], next_scaled])

                # Inverse transform forecast
                forecast_prices = scaler.inverse_transform(
                    np.concatenate((np.zeros((len(forecast_scaled), data.shape[1] - 1)),
                                    np.array(forecast_scaled).reshape(-1, 1)), axis=1)
                )[:, -1]
                forecast_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=n_days)

                # Plot historical + forecast
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Historical"))
                fig2.add_trace(go.Scatter(x=forecast_dates, y=forecast_prices, mode="lines+markers", name="Forecast"))
                st.plotly_chart(fig2, use_container_width=True)

                st.success(f"✅ Predicted next {n_days} days using {best_model_name}")

                # -----------------------
                # 6) Sentiment Analysis (Bonus)
                # -----------------------
                st.subheader("📰 Market Sentiment (Bonus Feature)")
                try:
                    # This will download a small transformer model on first run (requires internet)
                    sentiment_pipeline = pipeline("sentiment-analysis")
                    example_news = f"{stock_symbol} stock shows volatility amid recent market trends."
                    sentiment = sentiment_pipeline(example_news)[0]
                    st.write(f"News: *{example_news}*")
                    st.write(f"Sentiment: **{sentiment['label']}** (score={sentiment['score']:.2f})")
                except Exception as e:
                    st.warning(f"Could not run sentiment pipeline: {e}")
                    st.info("If you want sentiment analysis, ensure 'transformers' and 'torch' are installed and internet is available.")

