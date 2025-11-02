import os
import random
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from urllib.parse import urlparse
from datetime import datetime, timezone

# 設置 CPU 核心限制 (減少過熱風險)
os.environ["OMP_NUM_THREADS"] = "4" 
os.environ["TF_NUM_INTRAOP_THREADS"] = "4"
os.environ["TF_NUM_INTEROP_THREADS"] = "4"

# ========= 隨機種子（可重現） =========
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# ========= 先載入 .env，再取環境變數 (修正載入路徑) =========
# 修正 1：從當前腳本所在的目錄載入 .env 檔案
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env") 

# 載入環境變數
DB_URL = os.getenv("DB_URL") or os.getenv("DATABASE_URL")
STEP_MIN = int(os.getenv("STEP_MINUTES", "1"))
WINDOW = int(os.getenv("WINDOW", "60"))
EPOCHS = int(os.getenv("EPOCHS", "500"))
BATCH = int(os.getenv("BATCH", "128"))

# 可選訓練期間參數
TRAIN_DAYS = os.getenv("TRAIN_DAYS")
TRAIN_START_UTC = os.getenv("TRAIN_START_UTC")
TRAIN_END_UTC = os.getenv("TRAIN_END_UTC")


# ----------------------------------------------------------------------

if not DB_URL:
    raise RuntimeError("DB_URL 未設定，請檢查 .env 檔案。")
u = urlparse(DB_URL)
print("Using DB_URL:", f"{u.scheme}://{u.username}:******@{u.hostname}:{u.port}{u.path}")
print(f"STEP_MIN={STEP_MIN}, WINDOW={WINDOW}, EPOCHS={EPOCHS}, BATCH={BATCH}")

# ========= 欄位名稱 =========
TIME_COL = "timestamp"
PWR_COL = "system_power_watt"
N_FEATURES = 4 # 4 個特徵 (power_w, hour, dayofweek, lag_24h)

# ========= SQL（讀取資料） =========
where_extra, params = [], {}
if TRAIN_DAYS and TRAIN_DAYS.isdigit():
    where_extra.append("(timestamp_utc)::timestamptz >= now() - interval :train_days")
    params["train_days"] = f"{int(TRAIN_DAYS)} days"
if TRAIN_START_UTC:
    where_extra.append("(timestamp_utc)::timestamptz >= :start_utc")
    params["start_utc"] = TRAIN_START_UTC
if TRAIN_END_UTC:
    where_extra.append("(timestamp_utc)::timestamptz <= :end_utc")
    params["end_utc"] = TRAIN_END_UTC

where_clause = (" AND " + " AND ".join(where_extra) + "\n") if where_extra else ""

SQL = f"""
SELECT
  (timestamp_utc)::timestamptz AS {TIME_COL},
  {PWR_COL}
FROM energy_cleaned
WHERE timestamp_utc IS NOT NULL
  AND {PWR_COL} IS NOT NULL
{where_clause}ORDER BY (timestamp_utc)::timestamptz
""".strip()
print("DEBUG SQL:\n", SQL)

# ========= 讀取資料並進行基礎清洗 =========
# *** 關鍵修正 3：修正 'create_create_engine' 打字錯誤 ***
engine = create_engine(DB_URL, pool_pre_ping=True)
df = pd.read_sql(text(SQL), engine, params=params, parse_dates=[TIME_COL])
if df.empty:
    raise RuntimeError("energy_cleaned 查無資料（符合條件為 0）。")

print(f"Loaded rows: {len(df)} | time span: {df[TIME_COL].min()} → {df[TIME_COL].max()}")

# ========= 時間對齊、補值與特徵工程 =========
df = df.set_index(TIME_COL).sort_index()
df = df.resample(f"{STEP_MIN}min").mean()
df[PWR_COL] = df[PWR_COL].ffill().bfill()
df = df.dropna(subset=[PWR_COL])

# 尖峰平滑（防止異常值影響訓練）
q1, q3 = df[PWR_COL].quantile([0.25, 0.75])
iqr = q3 - q1
lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
df[PWR_COL] = df[PWR_COL].clip(lower, upper)
df[PWR_COL] = df[PWR_COL].rolling(window=3, center=True, min_periods=1).median()


# >>>>>>>>>>> 特徵工程: Hour, DayofWeek, Day-Ahead Lag (N_FEATURES=4) <<<<<<<<<<<<
df.index = pd.to_datetime(df.index)
df['hour'] = df.index.hour.astype(float)
df['dayofweek'] = df.index.dayofweek.astype(float)

# 關鍵：加入 24 小時 Lag 特徵 (1440 分鐘)
LAG_MINUTES = 24 * 60 
df['power_lag_24h'] = df[PWR_COL].shift(LAG_MINUTES)
df = df.dropna()

features = df[[PWR_COL, 'hour', 'dayofweek', 'power_lag_24h']].astype(float).values
# >>>>>>>>>>> 特徵工程結束 <<<<<<<<<<<<


# ========= 轉 numpy + 正規化與建序列 =========
if len(features) < WINDOW + 2:
    # 自動調整 WINDOW 邏輯（用於適應當前數據量）
    new_window = max(1, int(len(features) * 0.7) - 2)
    if new_window < WINDOW:
        print(f"⚠️ 資料量 ({len(features)}) 不足，自動降低 WINDOW: {WINDOW} → {new_window}")
        WINDOW = new_window
    if len(features) < WINDOW + 2:
        raise RuntimeError(f"資料量不足（{len(features)} < WINDOW+2）請增加資料或降低 WINDOW。")

print(f"Total processed rows: {len(features)}")

# 切分數據，再進行 Scaling (避免資料洩漏)
split_idx = int(len(features) * 0.8)
features_tr, features_val = features[:split_idx], features[split_idx:]
print(f"Split index: {split_idx} / {len(features)}")

# 訓練 Scaler 時使用所有 N_FEATURES=4
scaler = MinMaxScaler()
scaler.fit(features_tr)
features_tr_scaled  = scaler.transform(features_tr)
features_val_scaled = scaler.transform(features_val)

def make_sequences(arr, window, n_features):
    X, y = [], []
    for i in range(len(arr) - window - 1):
        X.append(arr[i:i+window])
        y.append(arr[i+window, 0]) 
    return np.array(X).reshape(-1, window, n_features), np.array(y).reshape(-1, 1)

# 呼叫 make_sequences 時傳遞 N_FEATURES=4
X_tr, y_tr   = make_sequences(features_tr_scaled, WINDOW, N_FEATURES)
X_val, y_val = make_sequences(features_val_scaled, WINDOW, N_FEATURES)

if len(X_tr) == 0 or len(X_val) == 0:
    raise RuntimeError("切分後無法形成序列，請調整資料量或 WINDOW。")

print(f"Train/Val shapes: X_tr={X_tr.shape}, X_val={X_val.shape}")

# ========= 建模與訓練 (Stacked LSTM - N_FEATURES=4) =========
model = Sequential([
    LSTM(128, input_shape=(WINDOW, N_FEATURES), return_sequences=True), 
    Dropout(0.1),
    LSTM(64, return_sequences=False),
    Dropout(0.1),
    Dense(1)
])
model.compile(optimizer="adam", loss="mse")

# ========= 訓練（LR 調度 + EarlyStopping） =========
es  = EarlyStopping(patience=15, restore_best_weights=True)
rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-5, verbose=1)

history = model.fit(
    X_tr, y_tr,
    validation_data=(X_val, y_val),
    epochs=EPOCHS, batch_size=BATCH,
    callbacks=[es, rlr],
    verbose=1
)

best_epoch = int(np.argmin(history.history["val_loss"]) + 1)

# ========= 評估（反標準化）=========
y_pred = model.predict(X_val, verbose=0)

# 反標準化時必須創建包含所有 N_FEATURES=4 的臨時陣列
y_temp = np.zeros((len(y_val), N_FEATURES))
y_pred_temp = np.zeros((len(y_pred), N_FEATURES))

# 將原始 y 值和預測 y 值放入第 0 列 (power_w 的位置)
y_temp[:, 0] = y_val.flatten()
y_pred_temp[:, 0] = y_pred.flatten()

# 反向轉換時只取第 0 列 (power_w)
y_val_w = scaler.inverse_transform(y_temp)[:, 0]
y_pred_w = scaler.inverse_transform(y_pred_temp)[:, 0]

rmse = float(np.sqrt(mean_squared_error(y_val_w, y_pred_w)))
mape = float(mean_absolute_percentage_error(y_val_w, y_pred_w) * 100)
best_val_loss = float(np.min(history.history["val_loss"]))
try:
    current_lr = float(tf.keras.backend.get_value(model.optimizer.learning_rate))
except Exception:
    current_lr = 0.0

print(f"✅ 最佳 Epoch 編號：{best_epoch} / {EPOCHS}")
print(f"RMSE(W): {rmse:.2f} | MAPE(%): {mape:.2f} | best_val_loss: {best_val_loss:.6f} | lr={current_lr:.6g}")

# ========= 輸出模型與 Scaler =========
out_dir = Path("/models")
if not out_dir.exists():
    out_dir = Path(__file__).resolve().parents[1] / "models"
out_dir.mkdir(parents=True, exist_ok=True)

model_path  = out_dir / "lstm_carbon_model.keras"
scaler_path = out_dir / "scaler_power.pkl"
model.save(model_path)
joblib.dump(scaler, scaler_path)
print("✅ Saved:", model_path, "and", scaler_path)

# ========= 訓練日誌寫入 /models/train_log.txt =========
log_path = out_dir / "train_log.txt"
data_start = df.index.min()
data_end   = df.index.max()
n_raw      = len(df)
n_tr       = len(X_tr)
n_val      = len(X_val)

log_line = (
    f"{datetime.now(timezone.utc).isoformat()} "
    f"| best_epoch={best_epoch}/{EPOCHS} best_val_loss={best_val_loss:.6f} "
    f"| RMSE={rmse:.3f}W MAPE={mape:.2f}% lr={current_lr:.6g} "
    f"| window={WINDOW} step_min={STEP_MIN} batch={BATCH} "
    f"| data_span=[{data_start} → {data_end}] n_raw={n_raw} n_tr_seq={n_tr} n_val_seq={n_val} "
    f"| model={model_path.name} scaler={scaler_path.name}\n"
)

try:
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(log_line)
    print(f"📝 Appended train log → {log_path}")
except Exception as e:
    print(f"⚠️ Failed to write train log: {e}")