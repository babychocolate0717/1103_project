# 使用官方的 Python 基礎映像檔，選擇一個適合生產環境的版本
# 我們使用 slim 版本以減少最終映像檔大小
FROM python:3.10-slim

# 設定工作目錄
WORKDIR /app

# ----------------------------------------------------
# 1. 安裝系統依賴
# ----------------------------------------------------
# psycopg2 需要這些庫來編譯 PostgreSQL 驅動
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# ----------------------------------------------------
# 2. 拷貝依賴文件並安裝 Python 庫
# ----------------------------------------------------
# 假設您在 exporter/ 資料夾內有一個 requirements.txt 檔案
COPY requirements.txt .

# 安裝 Python 依賴
RUN pip install --no-cache-dir -r requirements.txt

# ----------------------------------------------------
# 3. 拷貝應用程式碼
# ----------------------------------------------------
# 拷貝您的 Exporter 程式碼到容器內
COPY exporter.py .

# ----------------------------------------------------
# 4. 運行服務
# ----------------------------------------------------

# 服務預設運行在 9188 port，這是 Prometheus Exporter 常用端口
EXPOSE 9188

# 使用 uvicorn 啟動 FastAPI 應用
# 這裡假設您使用 Gunicorn 來管理 Uvicorn worker (更適合生產環境)
# 並且使用 1-4 個 worker (視 CPU 核心數而定)
# 如果是簡單測試，可以只用： CMD ["uvicorn", "exporter:app", "--host", "0.0.0.0", "--port", "9188"]
CMD ["uvicorn", "exporter:app", "--host", "0.0.0.0", "--port", "9188"]