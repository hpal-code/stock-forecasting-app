📈 Stock Market Forecasting Web App

Deep learning–based stock price forecasting using LSTM, GRU & Transformer models with real-time stock data.

🚀 Tech Stack

Python

TensorFlow / Keras

Streamlit

yFinance

NumPy, Pandas

Plotly

HuggingFace Transformers

📌 Project Overview

This web app predicts future stock prices using Deep Learning architectures.
It also performs market sentiment analysis, fetches real-time stock data, and displays interactive graphs.

🧠 Features

✔ Real-time stock data
✔ LSTM prediction
✔ GRU prediction
✔ Transformer (Multi-Head Attention)
✔ Auto-select best model (RMSE/MAE)
✔ Market sentiment analysis
✔ Beautiful Streamlit dashboard

🛠 Workflow
1. Data Collection
import yfinance as yf

2. Preprocessing

Missing values handled

Scaling (MinMaxScaler)

Time-series windowing

3. Models Used

LSTM (2-layer)

GRU (2-layer)

Transformer (Multi-head attention)

4. Evaluation Metrics

RMSE

MAE

5. Visualization

Historical vs Predicted

Future Forecast

📊 Screenshots
![Banner](images/stock_prediction_banner.png)
![UI](images/stock_prediction_ui.png)

📦 Installation
pip install -r requirements.txt

▶️ Run App
streamlit run app.py

📁 Recommended Folder Structure
project/
│── app.py
│── model_training.py
│── model/
│    ├── lstm_model.h5
│    ├── gru_model.h5
│    ├── transformer_model.h5
│── images/
│    ├── stock_prediction_banner.png
│    ├── stock_prediction_ui.png
│── requirements.txt
│── README.md

✨ Author

Himanshu Pal
AI/ML Engineer | India
