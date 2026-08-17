# 多來源 AI 偵測平台（本機 macOS 部署）

一個基於 YOLO 的多路影像分析平台，在 macOS（Apple Silicon）本機運行，使用 **MPS（GPU）** 推論 + **VideoToolbox** 硬體編解碼，同時接入多路 RTSP / FLV / HLS / MSE 串流，做即時偵測、骨架、姿勢、跌倒判斷、偵測區域與紅線計數，並把標註結果重新以 RTSP 推送出去。

---

## 目錄

1. [功能一覽](#功能一覽)
2. [硬體加速說明（MPS）](#硬體加速說明mps)
3. [首次安裝](#首次安裝)
4. [啟動 / 停止 / 重啟](#啟動--停止--重啟)
5. [Web 介面完整操作](#web-介面完整操作)
6. [RTSP 輸出](#rtsp-輸出)
7. [內建模型](#內建模型)
8. [支援的來源格式](#支援的來源格式)
9. [設定檔與環境變數](#設定檔與環境變數)
10. [疑難排解](#疑難排解)

---

## 功能一覽

| 功能 | 說明 |
|------|------|
| 多來源 | 同時分析多路串流，每路獨立 |
| 每路獨立模型 | 每個來源可選不同 YOLO 模型 |
| 類別選擇 | 每個來源可勾選要偵測的類別 |
| 自訂模型上傳 | 上傳自己的 `.pt` 模型 |
| 偵測區域 | 畫多邊形，只偵測區域內物件 |
| 紅線計數 | 畫線，物件跨線累計 +1 |
| 骨架 + 姿勢 | 站立 / 躺著 / 跌倒判斷 |
| 啟動 / 停止 | 單路或全部的啟動停止 |
| TV Wall | 一次看全部來源畫面 |
| 預覽 | 獨立視窗看單路即時畫面 |
| RTSP 重廣播 | 標註結果重新推送 |

---

## 硬體加速說明（MPS）

- **MPS = Metal Performance Shaders**，Apple 晶片的 GPU 運算框架。
- 本機跑會自動用 **MPS（GPU）推論** + **VideoToolbox（硬體解碼/編碼）**，速度遠快於 CPU。
- 啟動時會印出偵測結果，正常應顯示：
  ```
  [init] device=mps hwaccel=videotoolbox encoder=h264_videotoolbox ...
  ```
  看到 `device=mps` 就是吃到 GPU 加速。

---

## 首次安裝

**環境：** macOS（Apple Silicon 建議）、Python 3.12、Homebrew

```bash
# 1) 進入專案目錄
cd "你的專案路徑"

# 2) 建立虛擬環境
python3 -m venv venv

# 3) 安裝依賴
venv/bin/pip install -r requirements.txt

# 4) 安裝 MediaMTX（RTSP 伺服器）
brew install mediamtx

# 5) 讓腳本可執行
chmod +x start.sh stop.sh
```

---

## 啟動 / 停止 / 重啟

### 啟動
```bash
./start.sh
```
啟動後：
- 管理介面：`http://localhost:8000`
- TV Wall：`http://localhost:8000/tvwall`

### 停止
```bash
./stop.sh
```
會停止偵測服務 + MediaMTX + 所有 ffmpeg 子進程。

### 重啟
```bash
./stop.sh && ./start.sh
```

### 查看是否在運行
```bash
pgrep -fl "serve_vehicles|mediamtx"
```

### 查看日誌
```bash
tail -f /tmp/serve.log        # 偵測服務日誌（含 FPS）
tail -f /tmp/mediamtx.log     # RTSP 伺服器日誌
```

---

## Web 介面完整操作

### 主頁 `/`（管理來源）

**加入來源：**
1. 填「來源名稱」（英文/數字，例如 `cam1`）
2. 填「來源 URL」（見[支援格式](#支援的來源格式)）
3. 按「＋ 加入來源」

**每路操作按鈕：**
| 按鈕 | 功能 |
|------|------|
| 模型下拉 | 切換該路使用的模型 |
| 預覽 | 開獨立視窗看即時畫面 |
| 設定區域/紅線 | 進入該路設定頁 |
| ⏹ 停止 / ▶ 啟動 | 單路啟動停止（名稱前綠點=運行中） |
| 移除 | 刪除該來源 |

**頁面頂部：**
| 按鈕 | 功能 |
|------|------|
| 📺 TV Wall | 一次看全部畫面 |
| ▶ 啟動全部 | 啟動所有來源 |
| ⏹ 停止全部 | 停止所有來源 |

**上傳自訂模型：** 選 `.pt` 檔 → 按「上傳模型」，上傳後會出現在模型下拉。

### 設定頁 `/view/<名稱>`

進入後可設定該路的偵測方式：

1. **模型（model）** — 切換模型（車輛/頭盔/跌倒/骨架）
2. **imgsz** — 推論解析度，384（最快）～1280（最準）
3. **類別選擇** — 按「類別選擇」彈出勾選列，勾完按「套用類別」
4. **＋ 偵測區域** — 點多點畫多邊形，雙擊或「完成區域」結束（可畫多個）
5. **＋ 紅線(計數)** — 點兩下畫一條線（可畫多條）
6. **儲存** — 把區域與紅線套用生效
7. **重置計數** — 紅線累計歸零
8. **清除全部** — 移除所有區域與紅線

### 畫偵測區域（detection zone）
- 只有「中心點落在區域內」的物件才會被偵測，區域外完全忽略。
- 畫面顯示黃色半透明多邊形 + `Z1`、`Z2` 標籤。

### 畫紅線（counting line）
- 物件（車/人）**跨過紅線**即累計 +1（雙向都算）。
- 用 ByteTrack 追蹤 ID 判斷穿越，避免重複計。
- 畫面顯示紅色線 + `L1: N` 累計數。
- 左上角 `Total Cross` 為所有紅線的總穿越數。

### TV Wall `/tvwall`
- 所有來源自動排成網格（`ceil(√N)` 行×列）
- 點任一格跳到該路設定頁
- 「重新整理」按鈕可刷新（來源增減後用）

---

## RTSP 輸出

每個來源的標註結果會重新編碼成 H264 並推送到 MediaMTX：

| 用途 | URL |
|------|-----|
| 本機 | `rtsp://localhost:8554/<來源名稱>` |
| 其他裝置（同網段） | `rtsp://<本機IP>:8554/<來源名稱>` |
| HLS | `http://<本機IP>:8888/<來源名稱>/index.m3u8` |

> 查本機 IP：`ipconfig getifaddr en0`
>
> 例：來源名 `cam1`，本機 IP `192.168.1.132`，用 VLC 開 `rtsp://192.168.1.132:8554/cam1`

---

## 內建模型

| 模型檔 | 用途 | 類別 |
|--------|------|------|
| `yolo11n.pt` | 車輛（最快） | car / truck / bus / motorcycle / bicycle |
| `yolo11s.pt` | 車輛（預設） | 同上 |
| `helmet.pt` | 頭盔 | Hardhat / NO-Hardhat |
| `fall_detect.pt` | 跌倒偵測 | fallen / sitting / standing |
| `yolo11n-pose.pt` | 骨架（快） | person + 17 關鍵點 |
| `yolo11s-pose.pt` | 骨架（準） | person + 17 關鍵點 |

- 骨架模型會自動判斷 **standing（站立）/ lying（躺著）/ falling（跌倒）** 顯示在畫面左上。
- 官方 `yolo11*` 模型首次使用會自動下載；`helmet.pt`、`fall_detect.pt` 已放在專案目錄。

---

## 支援的來源格式

| 格式 | 範例 |
|------|------|
| RTSP | `rtsp://admin:pass@host:554/Streaming/Channels/101` |
| HLS | `http://host/live/index.m3u8` |
| FLV | `http://host/live.flv` 或 `rtmp://host/live` |
| MSE / fMP4 | `http://host/api/stream.mp4?src=NAME` |
| go2rtc 播放器頁 | `http://host/stream.html?src=NAME&mode=mse`（自動轉換） |
| WebSocket MSE | `ws://host/api/ws?src=NAME`（自動轉換） |
| MJPEG | `http://host/api/stream.mjpeg?src=NAME` |

---

## 設定檔與環境變數

### 設定檔
- 來源、模型、區域、紅線、計數全部**自動持久化**在 `sources_config.json`，重啟不會遺失。
- 上傳的自訂模型放在 `models/` 目錄。

### 環境變數（可選，通常不用改）
| 變數 | 預設 | 說明 |
|------|------|------|
| `FRAME_W` / `FRAME_H` | 1280 / 720 | 輸出解析度 |
| `IMGSZ` | 480 | 推論解析度 |
| `MODEL` | yolo11s.pt | 新來源預設模型 |
| `RTSP_HOST` | localhost:8554 | MediaMTX 位址 |
| `HTTP_PORT` | 8000 | Web 介面埠 |
| `CONFIG_FILE` | sources_config.json | 設定檔路徑 |
| `MODELS_DIR` | models | 上傳模型目錄 |

---

## 疑難排解

| 問題 | 處理 |
|------|------|
| FPS 很低 / 掉到 0.x | 降 imgsz（384/480）、用較小模型、減少來源數 |
| 多路同時跑崩潰 | 已用 MPS 鎖避免並行崩潰；若仍崩，減少來源數 |
| MSE 來源沒畫面 | URL 要用 `api/stream.mp4`，不是 `stream.html` 網頁 |
| 預覽/TV Wall 黑色 | 重啟後等 10–20 秒暖機；確認來源有在推流 |
| 回列表再進 TV Wall 顯示不出 | 已修正連線殘留；重連瀏覽器即可 |
| RTSP 輸出 404 | 確認 MediaMTX 有啟動（`pgrep -f mediamtx`） |
| 啟動後 `device=cpu` | 表示沒吃到 GPU，確認在 Apple Silicon 本機跑 |

---

## Docker（選用，暫不建議在 Mac 用）

> Mac 上的 Docker 無法存取 MPS（GPU），只能跑 CPU（約 6 FPS）。**本機請用 `./start.sh`。**

若日後要部署到 Linux + NVIDIA GPU 伺服器：
```bash
docker compose up -d --build     # 啟動（伺服器會自動用 CUDA）
docker compose down              # 停止
```
