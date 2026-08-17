#!/bin/bash
# ============================================================
# 啟動整個 YOLO 偵測方案（MediaMTX RTSP 伺服器 + 偵測服務）
# ============================================================
set -e
cd "$(dirname "$0")"

PYTHON="venv/bin/python"
MTX_BIN="$(command -v mediamtx || echo /opt/homebrew/opt/mediamtx/bin/mediamtx)"
MTX_CFG="/opt/homebrew/etc/mediamtx/mediamtx.yml"

# 1) 啟動 MediaMTX RTSP 伺服器
if pgrep -f mediamtx > /dev/null 2>&1; then
  echo "[OK] MediaMTX 已在運行"
else
  if [ -x "$MTX_BIN" ]; then
    nohup "$MTX_BIN" "$MTX_CFG" > /tmp/mediamtx.log 2>&1 &
    echo "[OK] MediaMTX 已啟動 (RTSP :8554 / HLS :8888)"
  else
    echo "[WARN] 找不到 mediamtx，跳過 RTSP 伺服器（RTSP 輸出不可用）"
  fi
fi

# 2) 啟動偵測服務
if [ ! -x "$PYTHON" ]; then
  echo "[ERROR] 找不到 venv，請先執行安裝步驟 (python3 -m venv venv && venv/bin/pip install -r requirements.txt)"
  exit 1
fi
if pgrep -f serve_vehicles.py > /dev/null 2>&1; then
  echo "[OK] 偵測服務已在運行"
else
  nohup "$PYTHON" serve_vehicles.py > /tmp/serve.log 2>&1 &
  echo "[OK] 偵測服務已啟動 (Web UI http://localhost:8000)"
fi

echo ""
echo "  管理介面  : http://localhost:8000"
echo "  TV Wall   : http://localhost:8000/tvwall"
echo "  停止      : ./stop.sh"
