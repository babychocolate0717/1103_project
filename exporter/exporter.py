from fastapi import FastAPI, Response
from prometheus_client import Gauge, generate_latest
import psycopg2
from urllib.parse import urlparse
import sys

# ----------------------------------------------------------------------
# 1. 資料庫連線配置
# ----------------------------------------------------------------------
DATABASE_URL = "postgresql+psycopg2://user:password@db:5432/energy"

def parse_db_url(url):
    """將標準 DB URL 解析成 psycopg2 連線所需的關鍵字參數"""
    parsed_url = urlparse(url)
    return {
        'host': parsed_url.hostname,
        'port': parsed_url.port,
        'database': parsed_url.path.lstrip('/'),
        'user': parsed_url.username,
        'password': parsed_url.password
    }

DB_PARAMS = parse_db_url(DATABASE_URL)

# ----------------------------------------------------------------------
# 2. 應用程式初始化、輔助函數與 Prometheus 指標定義
# ----------------------------------------------------------------------
app = FastAPI()

# 輔助函數：將策略文字標籤轉換為數值代碼
def map_strategy_to_code(label):
    """將策略標籤映射為數值代碼"""
    label = label.upper()
    if label == 'HIGH':
        return 3.0
    elif label == 'MID':
        return 2.0
    elif label == 'LOW':
        return 1.0
    return 0.0

# --- LSTM 預測指標 (新增 device_id 和 location 標籤) ---
PREDICTED_WATT_GAUGE = Gauge('lstm_predicted_watt', 
                             'LSTM模型預測的實時功耗值 (Watt)',
                             ['device_id', 'location'])

STRATEGY_LEVEL_GAUGE = Gauge('ai_strategy_level', 
                             'AI策略建議的數值代碼 (1-LOW, 3-HIGH)', 
                             ['strategy_label', 'confidence_score', 'device_id', 'location'])

# --- 系統資訊指標 (新增 device_id 標籤) ---
SYSTEM_CPU_WATT_GAUGE = Gauge('system_cpu_power_watt', 'CPU 實時功耗 (W)', ['device_id'])
SYSTEM_GPU_WATT_GAUGE = Gauge('system_gpu_power_watt', 'GPU 實時功耗 (W)', ['device_id'])
SYSTEM_MEMORY_MB_GAUGE = Gauge('system_memory_used_mb', '系統記憶體使用量 (MB)', ['device_id'])




def collect_metrics_from_db():
    """從 PostgreSQL 資料庫抓取所有裝置的最新 ML 和系統指標數據並更新 Prometheus 指標"""
    conn = None
    
    # 清除所有舊的指標值，確保只導出最新的數據
    PREDICTED_WATT_GAUGE.clear()
    STRATEGY_LEVEL_GAUGE.clear()
    SYSTEM_CPU_WATT_GAUGE.clear()
    SYSTEM_GPU_WATT_GAUGE.clear()
    SYSTEM_MEMORY_MB_GAUGE.clear()
    
    # 初始化安全變數
    device_id_safe = 'UNKNOWN_DEVICE'
    location_safe = 'UNKNOWN_LOCATION'
    
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        cursor = conn.cursor()

        # ----------------------------------------------------
        # 1. 獲取所有活躍裝置 ID 列表 (從 carbon_emissions 表中獲取)
        # ----------------------------------------------------
        active_devices_sql = """
            SELECT DISTINCT (recommended_strategy ->> 'device_id') AS device_id
            FROM carbon_emissions
            WHERE (recommended_strategy ->> 'device_id') IS NOT NULL;
        """
        cursor.execute(active_devices_sql)
        device_list = [row[0] for row in cursor.fetchall()] 
        
        # ----------------------------------------------------
        # 2. 遍歷所有裝置，針對性地執行查詢和導出 (核心邏輯)
        # ----------------------------------------------------
        
        for device_id in device_list:
            
            # --- 確保 device_id_safe 在迴圈內被賦予值 ---
            device_id_safe = device_id if device_id else 'UNKNOWN_DEVICE'
            
            # --- 2A. 查詢最新的 Location (從 energy_cleaned) ---
            location_sql = f"""
                SELECT location
                FROM energy_cleaned
                WHERE device_id = '{device_id_safe}' AND location IS NOT NULL
                ORDER BY timestamp_utc DESC
                LIMIT 1;
            """
            cursor.execute(location_sql)
            location_result = cursor.fetchone()
            location_safe = location_result[0] if location_result else 'Unknown Location'
            
            # --- 2B. 查詢最新的 ML 預測和策略 (針對單一裝置 ID) ---
            ml_sql_query = f"""
            SELECT
                COALESCE(predicted_power_w, 0.0),
                recommended_strategy ->> 'load_level',
                COALESCE(recommended_strategy ->> 'confidence', '0.0')
            FROM carbon_emissions
            WHERE predicted_power_w IS NOT NULL
              AND recommended_strategy ->> 'device_id' = '{device_id_safe}' 
            ORDER BY timestamp_to DESC
            LIMIT 1;
            """
            cursor.execute(ml_sql_query)
            ml_record = cursor.fetchone()

            # --- 2C. 查詢最新的系統功耗資訊 (針對單一裝置 ID) ---
            system_sql_query = f"""
            SELECT
                COALESCE(cpu_power_watt, 0.0), COALESCE(gpu_power_watt, 0.0), COALESCE(memory_used_mb, 0.0)
            FROM energy_cleaned
            WHERE device_id = '{device_id_safe}' 
            ORDER BY timestamp_utc DESC
            LIMIT 1;
            """
            cursor.execute(system_sql_query)
            system_record = cursor.fetchone()

            # ----------------------------------------------------
            # 3. 數據處理與指標更新 (強制導出每一個裝置的指標)
            # ----------------------------------------------------

            # A. 更新 LSTM/策略指標
            if ml_record:
                predicted_w_value, level_label, confidence_str = ml_record
                
                predicted_w_float = float(predicted_w_value)
                level_label_safe = level_label.strip() if level_label else 'UNKNOWN'
                confidence_safe = confidence_str.strip() if confidence_str else '0.0'
                
                def map_strategy_to_code(label):
                    if label == 'LOW': return 1.0
                    if label == 'MID': return 2.0
                    if label == 'HIGH': return 3.0
                    return 0.0
                    
                level_code_float = map_strategy_to_code(level_label_safe)

                # 1. 更新功耗預測指標
                PREDICTED_WATT_GAUGE.labels(device_id=device_id_safe, location=location_safe).set(predicted_w_float)
                
                # 2. 更新策略等級指標
                STRATEGY_LEVEL_GAUGE.labels(
                    strategy_label=level_label_safe,
                    confidence_score=confidence_safe,
                    device_id=device_id_safe, 
                    location=location_safe
                ).set(level_code_float)
            
            # B. 系統資訊指標更新
            if system_record:
                cpu_w, gpu_w, mem_mb = system_record
                
                SYSTEM_CPU_WATT_GAUGE.labels(device_id=device_id_safe).set(cpu_w)
                SYSTEM_GPU_WATT_GAUGE.labels(device_id=device_id_safe).set(gpu_w)
                SYSTEM_MEMORY_MB_GAUGE.labels(device_id=device_id_safe).set(mem_mb)

    except Exception as e:
        print(f"FATAL EXPORTER ERROR: {e}", file=sys.stderr)
        PREDICTED_WATT_GAUGE.labels(device_id='UNKNOWN_DEVICE', location='Error').set(0.0)
    finally:
        if conn:
            conn.close()

# 3. Prometheus 抓取端點
# ----------------------------------------------------------------------
@app.get("/metrics")
async def metrics_endpoint():
    collect_metrics_from_db()
    return Response(content=generate_latest(), media_type="text/plain")

# 運行指令範例：uvicorn exporter:app --host 0.0.0.0 --port 9188