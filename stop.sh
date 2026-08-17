#!/bin/bash
# ============================================================
# 停止整個 YOLO 偵測方案（偵測服務 + MediaMTX + ffmpeg 子進程）
# ============================================================
echo "停止偵測服務 ..."
pkill -f serve_vehicles.py 2>/dev/null && echo "  ✓ serve_vehicles 已停止" || echo "  - serve_vehicles 未在運行"

echo "停止 MediaMTX RTSP 伺服器 ..."
pkill -f mediamtx 2>/dev/null && echo "  ✓ mediamtx 已停止" || echo "  - mediamtx 未在運行"

echo "停止 ffmpeg 子進程 ..."
pkill -f "ffmpeg.*autoid" 2>/dev/null && echo "  ✓ 解碼器已停止" || echo "  - 無解碼器"
pkill -f "ffmpeg.*stream.mp4" 2>/dev/null && echo "  ✓ MSE 解碼器已停止" || echo "  - 無 MSE 解碼器"
pkill -f "ffmpeg.*8554" 2>/dev/null && echo "  ✓ RTSP 編碼器已停止" || echo "  - 無 RTSP 編碼器"

sleep 1
REMAIN=$(pgrep -fl "serve_vehicles|mediamtx" 2>/dev/null | grep -v grep)
if [ -z "$REMAIN" ]; then
  echo ""
  echo "✅ 全部已停止"
else
  echo ""
  echo "⚠ 仍殘留進程："
  echo "$REMAIN"
fi
