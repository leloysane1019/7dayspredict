from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import joblib
import yfinance as yf
import pandas as pd
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.trend import MACD

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# モデル読み込み
model = joblib.load("xgb_model_7day.pkl")

# 学習時と同じ特徴量の列
feature_columns = [
    'Close', 'High', 'Low', 'Open', 'Volume',
    'RSI_14', 'MA_5', 'MA_25', 'MA_75',
    'Volume_MA_5', 'BB_bbm', 'BB_bbh', 'BB_bbl',
    'MACD', 'MACD_signal'
]

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "code": "9104.T",
        "prediction": None,
        "error": None
    })


@app.get("/predict", response_class=HTMLResponse)
async def predict(request: Request, code: str = "9104.T"):
    df = yf.download(code, period="90d")

    # --- ここで列名を固定化（銘柄コード入りを除去） ---
    df.columns = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

    if df.empty:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "code": code,
            "prediction": None,
            "error": "株価データを取得できませんでした。"
        })

    try:
        # テクニカル指標を追加
        df["RSI_14"] = RSIIndicator(close=df["Close"], window=14).rsi()
        df["MA_5"] = df["Close"].rolling(window=5).mean()
        df["MA_25"] = df["Close"].rolling(window=25).mean()
        df["MA_75"] = df["Close"].rolling(window=75).mean()
        df["Volume_MA_5"] = df["Volume"].rolling(window=5).mean()

        bb = BollingerBands(close=df["Close"], window=25, window_dev=2)
        df["BB_bbm"] = bb.bollinger_mavg()
        df["BB_bbh"] = bb.bollinger_hband()
        df["BB_bbl"] = bb.bollinger_lband()

        macd = MACD(close=df["Close"])
        df["MACD"] = macd.macd()
        df["MACD_signal"] = macd.macd_signal()

        df = df.dropna()

        # 最新データ1件で予測
        latest = df.iloc[[-1]]
        X_pred = latest[feature_columns]
        y_pred = model.predict(X_pred)[0]

        return templates.TemplateResponse("index.html", {
            "request": request,
            "code": code,
            "prediction": round(y_pred, 2),
            "error": None
        })

    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "code": code,
            "prediction": None,
            "error": f"予測中にエラーが発生しました: {e}"
        })