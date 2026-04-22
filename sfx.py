import pandas as pd,numpy as np, matplotlib.pyplot as plt, seaborn as sns, time
import ccxt
import talib as ta
from xgboost import XGBRegressor


def get_data(coin, tf):
    exc = ccxt.binance({
    "enableRateLimit": True,
    "timeout": 30000})
    data = exc.fetch_ohlcv(coin, tf)
    df = pd.DataFrame(data, columns=[
        'timestamp','open','high','low','close','volume'])
    df["datetime"] = (pd.to_datetime(df["timestamp"], unit='ms', utc=True)
    .dt.tz_convert('Asia/Kolkata'))
    df = df.drop("timestamp",axis=1)
    df.set_index("datetime",inplace=True)

    return df

def create_feature(df):
    df["ema_9"] = df["close"].ewm(span=9).mean()
    df["ema_21"] = df["close"].ewm(span=21).mean()
    df["ema_50"] = df["close"].ewm(span=50).mean()
    df["return"] = df["close"].pct_change()
    df["volatility_stat"] = df["return"].rolling(20).std()
    df["volatility_range"] = (df["high"] - df["low"]) / df["close"]
    df["volatility_ratio"] =df["volatility_range"] /df["volatility_stat"]
    df["rsi"] = ta.RSI(df["close"].values, timeperiod = 14)
    df["atr"] = ta.ATR(df["high"].values, df["low"].values, df["close"].values, timeperiod=14)
    df["vol_mean"] = df["volume"].rolling(20).mean()
    df["vol_std"] = df["volume"].rolling(20).std()
    df["volume_z"] = (df["volume"] - df["vol_mean"]) / df["vol_std"]
    df["volume_spike"] = (df["volume"] > df["vol_mean"] * 1.5).astype(int)
    df["resistance"] = df["high"].rolling(20).max()
    df["support"] = df["low"].rolling(20).min()
    df["dist_to_resistance"] = df["resistance"] - df["close"]
    df["dist_to_support"] = df["close"] - df["support"]
    df["dist_res_norm"] = df["dist_to_resistance"] / df["atr"]
    df["dist_sup_norm"] = df["dist_to_support"] / df["atr"]

    return df

def train_model(df, steps):
    df["future_close"] = df["close"].shift(-steps)
    features = ["ema_9", "ema_21", "ema_50", "return", "volatility_range", "volatility_ratio", "rsi", 
            "atr", "volume_z", "volume_spike", "dist_res_norm", "dist_sup_norm"]
    preds = []
    X = df[features]
    y = df["future_close"]
    X = X[:-steps]
    y = y[:-steps]
    X_f = X[-steps:]
    model = XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05)
    model.fit(X, y)
    y_f = model.predict(X_f)

    return y_f

def predict_future(df,steps):
    features = ["ema_9", "ema_21", "ema_50", "return", "volatility_range",
                "volatility_ratio", "rsi", "atr", "volume_z",
                "volume_spike", "dist_res_norm", "dist_sup_norm"]

    df_temp = df.copy()
    preds = []
    X = df[features]
    y = df["close"]
    model = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05)
    model.fit(X,y)

    for _ in range(steps):
        X_f = df_temp[features].iloc[-1:]
        pred = model.predict(X_f)[0]
        preds.append(pred)

        new_row = df_temp.iloc[-1:].copy()
        new_row["close"] = pred

        df_temp = pd.concat([df_temp, new_row])
        df_temp = create_feature(df_temp)

    return np.array(preds)






    


