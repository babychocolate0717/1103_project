import os, asyncio, datetime as dt
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse
import socket

import numpy as np
import pandas as pd
import joblib
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from sqlalchemy import create_engine, text, delete
from tensorflow.keras.models import load_model
from pydantic import BaseModel
from sqlalchemy.exc import OperationalError

# ---------- env & config ----------
load_dotenv()

# 核心設定
DATABASE_URL = os.getenv("DATABASE_URL") or os.getenv("DB_URL")
LOOKBACK_MIN = int(os.getenv("BATCH_LOOKBACK_MINUTES", "1440"))
STEP_MIN     = int(os.getenv("STEP_MINUTES", "1"))
MODEL_VERSION= os.getenv("MODEL_VERSION", "lstm_v1")
RUN_INTERVAL = int(os.getenv("RUN_INTERVAL_SECONDS", "60"))
EF           = float(os.getenv("EF", "0.474")) # 修正：使用您日誌中的 EF 值
WINDOW       = int(os.getenv("WINDOW", "60"))
DELTA_HR     = STEP_MIN / 60.0
N_FEATURES = 4
LAG_MINUTES = 24 * 60

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL / DB_URL 未設定")

# ----------------------------------------------------------------------------------
# *** 關鍵修正 A：解決 主機環境 無法識別 'db' 的問題 ***
# ----------------------------------------------------------------------------------
if 'db:5432' in DATABASE_URL:
    try:
        socket.gethostbyname('db')
        print("INFO: Running inside Docker, 'db' hostname is resolvable.")
    except socket.error:
        DATABASE_URL = DATABASE_URL.replace("db:5432", "localhost:5433")
        print(f"⚠️ Swapping DB host to {DATABASE_URL.split('@')[-1]} for host environment.")
# ----------------------------------------------------------------------------------

u = urlparse(DATABASE_URL)
print("Using DATABASE_URL:", f"{u.scheme}://{u.username}:******@{u.hostname}:{u.port}{u.path}")
print(f"STEP_MIN={STEP_MIN}, WINDOW={WINDOW}, LOOKBACK_MIN={LOOKBACK_MIN}, EF={EF}, MODEL_VERSION={MODEL_VERSION}")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)

model = None
scaler = None

# --- Pydantic Model (API 用的) ---
class PredictMetric(BaseModel):
    timestamp: dt.datetime
    predicted_power_w: float 
    predicted_co2_kg: float 
    strategy: Dict[str, Any]

class FuturePrediction(BaseModel):
    timestamp: dt.datetime
    predicted_w: float
    predicted_co2: float

class FuturePredictionResponse(BaseModel):
    status: str
    steps: int
    predictions: List[FuturePrediction]

class GrafanaPredictionPoint(BaseModel):
    timestamp: dt.datetime
    predicted_w: float
    device: str

class GrafanaFutureResponse(BaseModel):
    status: str
    steps: int
    predictions: List[GrafanaPredictionPoint]

class DeviceInfo(BaseModel):
    device_id: str
    location: str

class DeviceListResponse(BaseModel):
    devices: List[DeviceInfo]


# --- 輔助函式 (Helpers) ---

def floor_to_step(dt_obj: dt.datetime, step_min: int) -> dt.datetime:
    if step_min <= 0: return dt_obj
    delta = dt.timedelta(minutes=dt_obj.minute % step_min,
                         seconds=dt_obj.second,
                         microseconds=dt_obj.microsecond)
    return (dt_obj - delta).replace(microsecond=0) # 確保微秒為 0

def get_power_thresholds() -> Dict[str, float]:
    return {"p20": 100.0, "p80": 400.0}

def recommend_strategy(pred_w: float, band_thresholds: dict, device_id: str, location: str) -> Dict[str, Any]:
    p80 = band_thresholds.get("p80", 400.0)
    p20 = band_thresholds.get("p20", 100.0)
    base_strategy = {"device_id": f"{device_id}", "location": f"{location}", "level_code": 2.0}
    if pred_w >= p80:
        return {**base_strategy, "load_level": "HIGH", "level_code": 3.0, "summary": "高負載預測：建議立即採取節能措施。", "recommendations": ["限制 GPU 功耗", "批次任務重新排程"]}
    elif pred_w <= p20:
        return {**base_strategy, "load_level": "LOW", "level_code": 1.0, "summary": "低功耗預測：適合執行耗時任務。", "recommendations": ["開始執行模型訓練或數據備份"]}
    else:
        return {**base_strategy, "load_level": "MID", "level_code": 2.0, "summary": "中等功耗預測：持續監控即可。", "recommendations": ["維持正常監控"]}


def ensure_pred_table(engine):
    """
    【重大修改】
    我們新增一個 'future_predictions' 資料表，專門存放未來 60 分鐘的預測。
    """
    sql_carbon_emissions = text("""
    CREATE TABLE IF NOT EXISTS carbon_emissions (
      timestamp_from     timestamptz NOT NULL,
      timestamp_to       timestamptz NOT NULL,
      horizon_steps      integer     NOT NULL,
      predicted_power_w  double precision NOT NULL,
      predicted_co2_kg   double precision NOT NULL,
      model_version      text        NOT NULL,
      recommended_strategy jsonb,
      created_at         timestamptz NOT NULL DEFAULT now(),
      device_id          text NOT NULL,
      PRIMARY KEY (timestamp_to, model_version, device_id)
    );
    """)
    
    # 【新增的資料表】
    sql_future_predictions = text("""
    CREATE TABLE IF NOT EXISTS future_predictions (
        timestamp           timestamptz NOT NULL,
        predicted_w         double precision NOT NULL,
        device_id           text NOT NULL,
        model_version       text NOT NULL,
        created_at          timestamptz NOT NULL DEFAULT now(),
        PRIMARY KEY (timestamp, device_id, model_version)
    );
    -- 為查詢優化建立索引
    CREATE INDEX IF NOT EXISTS idx_future_predictions_timestamp 
    ON future_predictions (timestamp DESC);
    """)

    with engine.begin() as conn:
        print("Ensuring table 'carbon_emissions'...")
        conn.execute(sql_carbon_emissions)
        print("Ensuring table 'future_predictions'...")
        conn.execute(sql_future_predictions)

def get_all_active_devices_labels():
    sql = text("""
        SELECT DISTINCT ON (device_id) device_id, location
        FROM energy_cleaned
        WHERE device_id IS NOT NULL AND location IS NOT NULL
        ORDER BY device_id, timestamp_utc DESC;
    """)
    with engine.connect() as conn:
        result = conn.execute(sql).fetchall()
        return [{"device_id": row[0], "location": row[1]} for row in result if row[0]]

def _load_model_sync():
    models_dir = Path(__file__).resolve().parents[1] / "models"
    keras_path = models_dir / "lstm_carbon_model.keras"
    h5_path    = models_dir / "lstm_carbon_model.h5"
    scaler_path= models_dir / "scaler_power.pkl"
    model_path = keras_path if keras_path.exists() else h5_path
    if not model_path.exists():
        raise FileNotFoundError(f"找不到模型檔：{keras_path} 或 {h5_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"找不到 scaler 檔：{scaler_path}")
    print(f"Loading model: {model_path.name}")
    loaded_model = load_model(model_path, compile=False)
    loaded_scaler = joblib.load(scaler_path)
    return loaded_model, loaded_scaler


# ---------- data access & preprocessing (4 特徵輸入) ----------

def fetch_power_series(end_ts: dt.datetime, minutes_to_fetch: int, device_id: str) -> pd.DataFrame:
    """
    【修正版】
    抓取 `minutes_to_fetch` 筆 1 分鐘間隔的數據，並計算所需的 Lag 特徵。
    """
    if model is None or scaler is None:
        raise RuntimeError("Model service is not fully initialized yet.")
        
    # 我們需要 (minutes_to_fetch) 筆數據 + (LAG_MINUTES) 筆數據來計算 lag
    required_minutes = minutes_to_fetch + LAG_MINUTES
    start_ts = end_ts - dt.timedelta(minutes=required_minutes)
    
    sql = text("""
        SELECT
          (timestamp_utc)::timestamptz AS ts,
          system_power_watt
        FROM energy_cleaned
        WHERE timestamp_utc IS NOT NULL
          AND system_power_watt IS NOT NULL
          AND device_id = :device_id_val
          AND (timestamp_utc)::timestamptz >  :start
          AND (timestamp_utc)::timestamptz <= :end
        ORDER BY (timestamp_utc)::timestamptz
    """)

    with engine.connect() as conn:
        params = {"start": start_ts, "end": end_ts, "device_id_val": device_id}
        raw = pd.read_sql(sql, conn, params=params, parse_dates=["ts"])

    if raw.empty:
        return raw.rename(columns={"ts": "timestamp"})

    df = raw.rename(columns={"ts": "timestamp"}).set_index("timestamp").sort_index()
    rule = f"{STEP_MIN}min"
    
    # 建立完整的 1 分鐘時間索引
    full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq=rule)
    if full_index.empty:
        # 如果原始數據為空或只有一個點，reindex 會失敗
        return pd.DataFrame(columns=['timestamp', 'system_power_watt', 'hour', 'dayofweek', 'power_lag_24h'])
        
    df = df.reindex(full_index)

    df["system_power_watt"] = df["system_power_watt"].interpolate(method='time') # 內插
    df["system_power_watt"] = df["system_power_watt"].ffill().bfill() # 向前/向後填充
    
    df = df.dropna(subset=["system_power_watt"]).reset_index().rename(columns={"index": "timestamp"})
    
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['power_lag_24h'] = df['system_power_watt'].shift(LAG_MINUTES)
    
    # 填充 Lag 缺失值
    df['power_lag_24h'] = df['power_lag_24h'].bfill().ffill() 

    df = df.dropna(subset=['power_lag_24h'])
    
    if df.empty:
        return df

    df['timestamp'] = df['timestamp'].dt.floor('s') # 確保微秒為 0
    df['system_power_watt'] = df['system_power_watt'].astype(float)
    df['hour'] = df['hour'].astype(float)
    df['dayofweek'] = df['dayofweek'].astype(float)
    df['power_lag_24h'] = df['power_lag_24h'].astype(float)

    # 我們只需要回傳最後 `minutes_to_fetch` 筆數據
    return df[['timestamp', 'system_power_watt', 'hour', 'dayofweek', 'power_lag_24h']].tail(minutes_to_fetch)

def predict_next_power_w(df: pd.DataFrame) -> float:
    # (此函數與您原本的相同，不需修改)
    if model is None or scaler is None:
        raise RuntimeError("Model service is not fully initialized yet.")
    features = df[['system_power_watt', 'hour', 'dayofweek', 'power_lag_24h']].values.astype(float)
    if len(features) < WINDOW:
        raise ValueError(f"需要至少 {WINDOW} 筆資料（當前 {len(features)}），請增加 LOOKBACK_MIN 或降低 WINDOW。")
    last_window = features[-WINDOW:]
    last_scaled = scaler.transform(last_window).reshape(1, WINDOW, last_window.shape[1])
    y_scaled = model.predict(last_scaled, verbose=0)
    temp_array = np.zeros((1, scaler.n_features_in_))
    temp_array[:, 0] = y_scaled[:, 0]
    y_watt = scaler.inverse_transform(temp_array)[:, 0][0]
    return float(y_watt)

def upsert_carbon_emission(ts_from, ts_to, steps, pw, co2, strategy: Dict[str, Any], device_id: str):
    # (此函數與您原本的相同，不需修改)
    strategy_json = json.dumps(strategy)
    sql = text("""
        INSERT INTO carbon_emissions
        (timestamp_from, timestamp_to, horizon_steps, predicted_power_w, predicted_co2_kg, model_version, recommended_strategy, device_id)
        VALUES (:from, :to, :h, :pw, :co2, :mv, :strategy, :device_id_val)
        ON CONFLICT (timestamp_to, model_version, device_id) DO UPDATE
        SET predicted_power_w = EXCLUDED.predicted_power_w,
            predicted_co2_kg  = EXCLUDED.predicted_co2_kg,
            recommended_strategy = EXCLUDED.recommended_strategy,
            timestamp_from    = EXCLUDED.timestamp_from,
            horizon_steps     = EXCLUDED.horizon_steps;
    """)
    with engine.begin() as conn:
        conn.execute(sql, {
            "from": ts_from, "to": ts_to, "h": steps,
            "pw": pw, "co2": co2, "mv": MODEL_VERSION,
            "strategy": strategy_json,
            "device_id_val": device_id
        })

def predict_multi_step(initial_df: pd.DataFrame, steps: int, device_id: str, location: str):
    """
    【修正版】
    預測未來 `steps` 步，並回傳一個包含 {timestamp, predicted_w, device} 的列表。
    """
    if model is None or scaler is None:
        raise RuntimeError("Model service is not fully initialized yet.")

    current_df = initial_df.copy()
    future_predictions_list = [] # 存放字典的列表
    feature_cols = ['system_power_watt', 'hour', 'dayofweek', 'power_lag_24h']

    if 'timestamp' in current_df.columns:
        current_df = current_df.set_index('timestamp').sort_index()

    if current_df.empty:
        print(f"[{dt.datetime.utcnow().isoformat()}Z] predict_multi_step (Device {device_id}) received empty initial_df.")
        return []

    last_ts = current_df.index[-1]

    for i in range(1, steps + 1):
        next_ts = floor_to_step(last_ts + dt.timedelta(minutes=STEP_MIN), STEP_MIN)

        features_window = current_df[feature_cols].values.astype(float)[-WINDOW:]

        if len(features_window) < WINDOW:
            print(f"[{next_ts.isoformat()}Z] predict_multi_step (Device {device_id}) ran out of data.")
            break

        last_scaled = scaler.transform(features_window).reshape(1, WINDOW, N_FEATURES)
        y_scaled = model.predict(last_scaled, verbose=0)

        temp_array = np.zeros((1, scaler.n_features_in_))
        temp_array[:, 0] = y_scaled[:, 0]
        pred_watt = scaler.inverse_transform(temp_array)[:, 0][0]

        # --- 查找 Lag 值 ---
        lag_ts = next_ts - dt.timedelta(minutes=LAG_MINUTES)
        lag_value = 0.0
        try:
            lag_value = current_df.loc[lag_ts, 'system_power_watt']
        except KeyError:
            lag_value = current_df['power_lag_24h'].iloc[-1]
            # print(f"Warning: Lag value for {lag_ts} not found. Using last known lag: {lag_value}")
        # --- 建立新行 ---
        new_row_data = {
            'system_power_watt': pred_watt,
            'hour': float(next_ts.hour),
            'dayofweek': float(next_ts.weekday()),
            'power_lag_24h': lag_value
        }
        new_df_row = pd.DataFrame(new_row_data, index=[next_ts])
        current_df = pd.concat([current_df, new_df_row])

        # ------------------------------------------------------------------
        # *** 關鍵錯誤修正：將 np.float64 轉換為 Python float ***
        # ------------------------------------------------------------------
        future_predictions_list.append({
            "timestamp": next_ts, 
            "predicted_w": float(pred_watt), # <--- 在這裡修正
            "device_id": device_id, 
            "model_version": MODEL_VERSION
        })
        # ------------------------------------------------------------------
        
        last_ts = next_ts

    return future_predictions_list

# ---------- 【新增】將預測寫入 DB 的函數 ----------
def upsert_future_predictions(predictions: List[Dict[str, Any]], device_id: str):
    """
    將 predict_multi_step 產生的預測列表，批量寫入 'future_predictions' 資料表。
    """
    if not predictions:
        return

    # 先刪除這個裝置舊的預測
    sql_delete = text("""
        DELETE FROM future_predictions 
        WHERE device_id = :device_id AND model_version = :model_version;
    """)
    
    # 準備批量插入
    sql_insert = text("""
        INSERT INTO future_predictions (timestamp, predicted_w, device_id, model_version)
        VALUES (:timestamp, :predicted_w, :device_id, :model_version)
        ON CONFLICT (timestamp, device_id, model_version) DO NOTHING;
    """)
    
    with engine.begin() as conn:
        # 步驟 1: 刪除舊資料
        conn.execute(sql_delete, {"device_id": device_id, "model_version": MODEL_VERSION})
        
        # 步驟 2: 批量插入新資料
        conn.execute(sql_insert, predictions) # 'predictions' 是一個字典列表，SQLAlchemy 會自動匹配
        
    print(f"[{dt.datetime.utcnow().isoformat()}Z] Wrote {len(predictions)} future predictions to DB for device {device_id}.")

# ---------- 【修改】background loop ----------
async def loop_job():
    """
    【重大修改】
    loop_job 現在會同時執行 1 分鐘預測 (寫入 carbon_emissions) 
    和 60 分鐘預測 (寫入 future_predictions)。
    """
    await asyncio.sleep(10) # 初始延遲，等待 DB 完全啟動
    print(f"[{dt.datetime.utcnow().isoformat()}Z] Background loop_job started.")
    
    while True:
        try:
            devices_list = await asyncio.to_thread(get_all_active_devices_labels)
        except Exception as e:
            print(f"[{dt.datetime.utcnow().isoformat()}Z] CRITICAL: Failed to fetch device list from DB. Error: {e}")
            await asyncio.sleep(RUN_INTERVAL * 5) # 如果資料庫連不上，等待 5 分鐘
            continue

        if not devices_list:
            print(f"[{dt.datetime.utcnow().isoformat()}Z] Warning: No active devices found.")
            await asyncio.sleep(RUN_INTERVAL)
            continue
            
        now_raw = dt.datetime.utcnow()
        
        # --- 任務 1： 1 分鐘預測 (for carbon_emissions) ---
        ts_to_1_min = floor_to_step(now_raw + dt.timedelta(minutes=STEP_MIN), STEP_MIN)
        ts_from_1_min = ts_to_1_min - dt.timedelta(minutes=STEP_MIN)
        
        # --- 任務 2： 60 分鐘預測 (for future_predictions) ---
        steps_future = 60
        minutes_to_fetch_future = WINDOW + steps_future # 總共需要 60 + 60 = 120 筆歷史資料

        for device_info in devices_list:
            device_id = device_info['device_id']
            location = device_info['location']
            
            # --- 執行 1 分鐘預測 (舊邏輯) ---
            try:
                # 抓取 1 天的數據來做 1 分鐘預測
                df_1_min = await asyncio.to_thread(fetch_power_series, ts_from_1_min, LOOKBACK_MIN, device_id)
                if df_1_min.empty:
                    print(f"[{ts_to_1_min.isoformat()}Z] Job error (Device {device_id}): No data in lookback window ({LOOKBACK_MIN} min).")
                else:
                    pred_power_w = await asyncio.to_thread(predict_next_power_w, df_1_min)
                    kWh = (pred_power_w / 1000.0) * DELTA_HR
                    co2 = kWh * EF
                    thresholds = get_power_thresholds()
                    strategy = recommend_strategy(pred_power_w, thresholds, device_id, location)
                    await asyncio.to_thread(upsert_carbon_emission, ts_from_1_min, ts_to_1_min, 1, pred_power_w, co2, strategy, device_id)
                    print(f"[{ts_to_1_min.isoformat()}Z] Pred={pred_power_w:.2f} W | kWh={kWh:.6f} | CO2={co2:.6f} kg | Device: {device_id} | Strategy: {strategy['summary']}")
            except Exception as e:
                print(f"[{ts_to_1_min.isoformat()}Z] (1-min) Job error (Device {device_id}): {repr(e)}")

            # --- 執行 60 分鐘預測 (新邏輯) ---
            try:
                # 抓取 120 筆資料 (WINDOW + steps)
                df_future = await asyncio.to_thread(fetch_power_series, now_raw, minutes_to_fetch_future, device_id)
                
                if len(df_future) < WINDOW:
                    print(f"[{now_raw.isoformat()}Z] (60-min) Skipping {device_id}: not enough historical data (need {WINDOW}, got {len(df_future)}).")
                    continue
                
                # 預測未來 60 步
                future_results_list = await asyncio.to_thread(predict_multi_step, df_future, steps_future, device_id, location)
                
                # 將這 60 筆數據寫入 'future_predictions' 資料表
                if future_results_list:
                    await asyncio.to_thread(upsert_future_predictions, future_results_list, device_id)
                
            except Exception as e:
                print(f"[{now_raw.isoformat()}Z] (60-min) Job error (Device {device_id}): {repr(e)}")

        await asyncio.sleep(RUN_INTERVAL) # 等待 60 秒

# ---------- FastAPI endpoints ----------
app = FastAPI(title="Prediction API (LSTM → Carbon)")

@app.on_event("startup")
async def on_startup():
    global model, scaler
    print("Application startup...")
    await asyncio.to_thread(ensure_pred_table, engine)
    try:
        model, scaler = await asyncio.to_thread(_load_model_sync)
        print("INFO: Model and Scaler loaded successfully.")
    except FileNotFoundError as e:
        print(f"FATAL: Model initialization failed: {e}")
    except Exception as e:
        print(f"FATAL: Unknown error during model loading: {e}")
    
    if model and scaler:
        print("Starting background loop_job...")
        asyncio.create_task(loop_job())
    else:
        print("ERROR: Model not loaded, background job will NOT start.")

@app.get("/health")
def health():
    return {
        "status": "ok" if model is not None else "model_unloaded",
        "model_version": MODEL_VERSION
    }

@app.get("/")
def read_root():
    return {"message": "LSTM Prediction API is running."}

# ----------------------------------------------------------------------
# 【/devices API endpoint (供 Grafana 變數使用)】
# ----------------------------------------------------------------------
@app.get("/devices", response_model=DeviceListResponse)
async def get_devices_for_variable():
    """
    專門為 Grafana 儀表板變數（下拉式選單）提供
    所有活躍裝置的 device_id 和 location 列表。
    """
    try:
        device_info_list = await asyncio.to_thread(get_all_active_devices_labels)
        return DeviceListResponse(devices=device_info_list)
    except Exception as e:
        print(f"Error in /devices endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch devices: {e}")

# ----------------------------------------------------------------------
# 【/predict-future API endpoint (已廢棄，但保留)】
# ----------------------------------------------------------------------
@app.get("/predict-future", response_model=GrafanaFutureResponse, include_in_schema=False)
async def predict_future_emissions_for_grafana(
    steps: int = Query(60, ge=1, le=1440),
    device_ids: List[str] = Query(None, alias="device_id", description="要查詢的 Device ID 列表 (可多選)")
):
    """
    此 API 已不再是主要方法，因為 loop_job 會自動將數據寫入 'future_predictions'。
    保留此 API 供手動測試，但 Grafana 不應再使用它。
    """
    print("WARNING: /predict-future endpoint was called. This is deprecated.")
    if not device_ids:
        return GrafanaFutureResponse(status="success", steps=steps, predictions=[])
    
    device_id = device_ids[0] # 只處理第一個
    location_map = {}
    try:
        device_info_list = await asyncio.to_thread(get_all_active_devices_labels)
        location_map = {dev['device_id']: dev['location'] for dev in device_info_list}
    except Exception: pass
    
    location = location_map.get(device_id, "Unknown")
    now_raw = dt.datetime.utcnow()
    minutes_to_fetch = WINDOW + steps

    all_predictions: List[GrafanaPredictionPoint] = []

    try:
        initial_df = await asyncio.to_thread(fetch_power_series, now_raw, minutes_to_fetch, device_id)
        if len(initial_df) < WINDOW:
            print(f"Skipping device {device_id}: not enough historical data.")
            return GrafanaFutureResponse(status="success", steps=steps, predictions=[])

        future_results_raw = await asyncio.to_thread(predict_multi_step, initial_df, steps, device_id, location)
        
        # 轉換為 Pydantic 模型
        for res in future_results_raw:
             all_predictions.append(
                 GrafanaPredictionPoint(
                     timestamp=res["timestamp"], # res["timestamp"] 已經是 datetime 物件
                     predicted_w=float(res["predicted_w"]), # 再次確保是 float
                     device=res["device_id"]
                 )
             )
    except Exception as e:
        print(f"Error processing device {device_id} for /predict-future: {e}")

    return GrafanaFutureResponse(
        status="success",
        steps=steps,
        predictions=all_predictions
    )
