import json
import math
import os
import queue
import select
import socket
import subprocess
import threading
import time
from contextlib import nullcontext
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse, parse_qs

import cv2
import numpy as np
from ultralytics import YOLO

# --- environment-configurable settings (Docker-friendly) ---
W, H = int(os.environ.get("FRAME_W", "1280")), int(os.environ.get("FRAME_H", "720"))
FRAME_SIZE = W * H * 3
CONFIG_FILE = os.environ.get("CONFIG_FILE", "sources_config.json")
RTSP_HOST = os.environ.get("RTSP_HOST", "localhost:8554")
HTTP_PORT = int(os.environ.get("HTTP_PORT", "8000"))
MODEL_NAME = os.environ.get("MODEL", "yolo11s.pt")
IMGSZ = int(os.environ.get("IMGSZ", "480"))
VEHICLE_CLASSES = [int(c) for c in os.environ.get("CLASSES", "1,2,3,5,7").split(",")]

# --- model catalog: display name -> default detection classes (None = all) ---
MODEL_CATALOG = {
    "yolo11n.pt":      {"label": "YOLO11 Nano (車輛)",   "classes": [1, 2, 3, 5, 7], "task": "detect"},
    "yolo11s.pt":      {"label": "YOLO11 Small (車輛)",  "classes": [1, 2, 3, 5, 7], "task": "detect"},
    "helmet.pt":       {"label": "頭盔偵測",              "classes": None, "task": "detect"},
    "fall_detect.pt":  {"label": "跌倒/坐/站偵測",         "classes": None, "task": "detect"},
    "yolo11n-pose.pt": {"label": "骨架 Pose Nano",       "classes": None, "task": "pose"},
    "yolo11s-pose.pt": {"label": "骨架 Pose Small",      "classes": None, "task": "pose"},
}

MODELS_DIR = os.environ.get("MODELS_DIR", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

model_names_cache = {}


def resolve_model_path(name):
    for p in (os.path.join(MODELS_DIR, name), name):
        if os.path.exists(p):
            return p
    return name


def get_model_info(name):
    """Return (names_dict, task) for a model, cached."""
    if name in model_names_cache:
        return model_names_cache[name]
    try:
        m = YOLO(resolve_model_path(name))
        info = ({int(k): v for k, v in m.names.items()}, getattr(m, "task", "detect"))
    except Exception:
        info = ({}, "detect")
    model_names_cache[name] = info
    return info


def scan_models_dir():
    for f in sorted(os.listdir(MODELS_DIR)):
        if f.endswith(".pt") and f not in MODEL_CATALOG:
            MODEL_CATALOG[f] = {"label": f"自訂: {f}", "classes": None, "task": "detect"}


def make_placeholder(text="CONNECTING..."):
    img = np.zeros((H, W, 3), np.uint8)
    cv2.putText(img, text, (W // 2 - 260, H // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 0), 3)
    ok, jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return jpeg.tobytes() if ok else None


placeholder_jpeg = make_placeholder()

lock = threading.Lock()
sources = {}   # name -> {"url", "model", "zones", "lines", "counters"}
workers = {}   # name -> SourceWorker


def detect_device():
    """Pick the best inference backend: MPS (mac) > CUDA (nvidia) > CPU."""
    try:
        import torch
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


DEVICE = detect_device()

# Serialize inference only for MPS (Metal crashes on concurrent access).
# CPU / CUDA can run inference concurrently across source threads.
infer_ctx = threading.Lock() if DEVICE == "mps" else nullcontext()

if DEVICE == "cpu":
    try:
        import torch
        torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "4")))
    except Exception:
        pass


def detect_codecs(device):
    """Pick ffmpeg decode/encode backends matching the inference device.

    Tie the codec hardware to the inference backend: if there is no usable GPU
    for inference, assume there is none for codecs either (avoids picking a
    compiled-in-but-unavailable encoder like nvenc on a CPU-only container).
    """
    if device == "mps":
        return "videotoolbox", ("h264_videotoolbox", ["-b:v", "4000k"])
    if device == "cuda":
        return "cuda", ("h264_nvenc", ["-b:v", "4000k"])
    return None, ("libx264", ["-preset", "ultrafast", "-tune", "zerolatency", "-b:v", "4000k"])


HWACCEL, (ENCODER, ENCODER_ARGS) = detect_codecs(DEVICE)

print(f"[init] device={DEVICE} hwaccel={HWACCEL} encoder={ENCODER} "
      f"model={MODEL_NAME} imgsz={IMGSZ} frame={W}x{H} port={HTTP_PORT} rtsp={RTSP_HOST}",
      flush=True)


def load_persisted():
    global sources
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE) as f:
                d = json.load(f)
            for name, cfg in d.get("sources", {}).items():
                sources[name] = {
                    "url": cfg.get("url", ""),
                    "model": cfg.get("model", MODEL_NAME),
                    "classes": cfg.get("classes", "default"),
                    "imgsz": cfg.get("imgsz", IMGSZ),
                    "enabled": cfg.get("enabled", True),
                    "zones": cfg.get("zones", []),
                    "lines": cfg.get("lines", []),
                    "counters": {int(k): v for k, v in cfg.get("counters", {}).items()},
                }
        except Exception:
            pass


def save_persisted():
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump({"sources": sources}, f)
    except Exception:
        pass


def normalize_url(url):
    """Rewrite common go2rtc player URLs to a stream ffmpeg can read."""
    u = url.strip()
    low = u.lower()
    # go2rtc web player page: stream.html?src=X&mode=mse -> api/stream.mp4?src=X
    if "stream.html" in low:
        parsed = urlparse(u)
        qs = parse_qs(parsed.query)
        src = qs.get("src", [""])[0]
        mode = (qs.get("mode", [""])[0]).lower()
        base = f"{parsed.scheme}://{parsed.netloc}"
        if not src:
            return u
        if mode == "mjpeg":
            return f"{base}/api/stream.mjpeg?src={src}"
        return f"{base}/api/stream.mp4?src={src}"
    # go2rtc WebSocket MSE: ws://host/api/ws?src=X -> http://host/api/stream.mp4?src=X
    if low.startswith("ws://") and "/api/ws" in low:
        return u.replace("ws://", "http://", 1).replace("/api/ws", "/api/stream.mp4", 1)
    if low.startswith("wss://") and "/api/ws" in low:
        return u.replace("wss://", "https://", 1).replace("/api/ws", "/api/stream.mp4", 1)
    return u


def start_ffmpeg(url):
    url = normalize_url(url)
    cmd = ["ffmpeg", "-v", "error"]
    if HWACCEL:
        cmd += ["-hwaccel", HWACCEL]
    if url.lower().startswith(("rtsp://", "rtsps://")):
        cmd += ["-rtsp_transport", "tcp"]
    cmd += [
        "-i", url,
        "-vf", f"scale={W}:{H},setsar=1",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)


def read_frame(proc, timeout=10.0):
    buf = bytearray(FRAME_SIZE)
    view = memoryview(buf)
    filled = 0
    while filled < FRAME_SIZE:
        r, _, _ = select.select([proc.stdout], [], [], timeout)
        if not r:
            return None
        n = proc.stdout.readinto(view[filled:])
        if not n:
            return None
        filled += n
    return np.frombuffer(buf, dtype=np.uint8).reshape(H, W, 3)


def start_encoder(name):
    cmd = [
        "ffmpeg", "-v", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", "25",
        "-i", "-",
        "-c:v", ENCODER, *ENCODER_ARGS, "-g", "50",
        "-f", "rtsp", "-rtsp_transport", "tcp",
        f"rtsp://{RTSP_HOST}/{name}",
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)


def point_in_polygon(x, y, poly):
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def line_side(px, py, line):
    x1, y1 = line[0]
    x2, y2 = line[1]
    return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)


def classify_posture(kp):
    """Classify posture from COCO keypoints (17x3). Returns standing/lying/falling/unknown."""
    def pt(i):
        if kp[i][2] < 0.3:
            return None
        return (kp[i][0], kp[i][1])

    ls, rs = pt(5), pt(6)
    lh, rh = pt(11), pt(12)
    if not (ls and rs and lh and rh):
        return "unknown"
    sh = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
    hp = ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2)
    dy = hp[1] - sh[1]
    dx = hp[0] - sh[0]
    ang = abs(math.degrees(math.atan2(dy, dx)))
    if ang > 60:
        return "standing"
    if ang < 30:
        return "lying"
    return "falling"


def draw_zones(frame, zones):
    for idx, z in enumerate(zones):
        pts = np.array(z, dtype=np.int32).reshape(-1, 1, 2)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], (0, 200, 255))
        frame[:] = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)
        cv2.polylines(frame, [pts], True, (0, 255, 255), 3)
        cx = int(np.mean([p[0] for p in z]))
        cy = int(np.mean([p[1] for p in z]))
        cv2.putText(frame, f"Z{idx + 1}", (cx - 15, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)


def draw_lines(frame, lines, counters):
    for idx, l in enumerate(lines):
        p1 = (int(l[0][0]), int(l[0][1]))
        p2 = (int(l[1][0]), int(l[1][1]))
        cv2.line(frame, p1, p2, (0, 0, 255), 4)
        mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
        cv2.putText(frame, f"L{idx + 1}: {counters.get(idx, 0)}", (mid[0] - 40, mid[1] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)


class SourceWorker(threading.Thread):
    def __init__(self, name):
        super().__init__(daemon=True)
        self.name = name
        self.running = True
        self.latest_jpeg = None
        self.track_side = {}
        self.track_seen = {}
        self.frame_no = 0
        self.proc = None
        self.enc = None
        self.model = None
        self.model_name = None
        self.model_classes = None
        self.enc_queue = queue.Queue(maxsize=3)

    def stop(self):
        self.running = False
        for p in (self.proc, self.enc):
            if p and p.poll() is None:
                try:
                    p.kill()
                except OSError:
                    pass

    def encoder_loop(self):
        enc = start_encoder(self.name)
        self.enc = enc
        while self.running:
            try:
                frame = self.enc_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                enc.stdin.write(frame.tobytes())
                enc.stdin.flush()
            except (BrokenPipeError, OSError):
                enc.kill()
                enc = start_encoder(self.name)
                self.enc = enc
        try:
            enc.kill()
        except OSError:
            pass

    def push_async(self, frame):
        try:
            self.enc_queue.put_nowait(frame)
        except queue.Full:
            pass

    def ensure_model(self):
        with lock:
            src = sources[self.name]
            want = src.get("model", MODEL_NAME)
            want_classes = src.get("classes", "default")
        if self.model is not None and want == self.model_name:
            return
        self.model = YOLO(resolve_model_path(want))
        self.model_name = want
        self.model_task = getattr(self.model, "task", "detect")
        default_classes = MODEL_CATALOG.get(want, {}).get("classes", None)
        self.model_classes = default_classes if want_classes == "default" else want_classes
        # reset tracking on model switch
        self.track_side.clear()
        self.track_seen.clear()
        print(f"[{self.name}] model switched -> {want} (task={self.model_task})", flush=True)

    def run(self):
        self.ensure_model()
        self.proc = start_ffmpeg(sources[self.name]["url"])
        threading.Thread(target=self.encoder_loop, daemon=True).start()
        prev = time.time()

        while self.running:
            timeout = 15.0 if self.frame_no == 0 else 3.0
            frame = read_frame(self.proc, timeout)
            if frame is None:
                if not self.running:
                    break
                self.proc.kill()
                self.proc = start_ffmpeg(sources[self.name]["url"])
                continue

            self.frame_no += 1

            self.ensure_model()

            with lock:
                zs = [list(z) for z in sources[self.name]["zones"]]
                ls = [list(l) for l in sources[self.name]["lines"]]
                imgsz = sources[self.name].get("imgsz", IMGSZ)

            with infer_ctx:
                if ls:
                    results = self.model.track(
                        source=frame,
                        classes=self.model_classes,
                        conf=0.35,
                        imgsz=imgsz,
                        device=DEVICE,
                        verbose=False,
                        persist=True,
                        tracker="bytetrack.yaml",
                    )
                else:
                    results = self.model.predict(
                        source=frame,
                        classes=self.model_classes,
                        conf=0.35,
                        imgsz=imgsz,
                        device=DEVICE,
                        verbose=False,
                    )
            r = results[0]

            if r.boxes is not None and len(r.boxes) and zs:
                xyxy = r.boxes.xyxy
                cx = (xyxy[:, 0] + xyxy[:, 2]) / 2.0
                cy = (xyxy[:, 1] + xyxy[:, 3]) / 2.0
                keep = [i for i in range(len(xyxy))
                        if any(point_in_polygon(float(cx[i]), float(cy[i]), z) for z in zs)]
                r.boxes = r.boxes[keep] if keep else None

            if r.boxes is not None and len(r.boxes) and ls:
                ids = r.boxes.id
                if ids is not None:
                    ids = [int(v) for v in ids.tolist()]
                    xyxy = r.boxes.xyxy.tolist()
                    for tid, box in zip(ids, xyxy):
                        cx = (box[0] + box[2]) / 2.0
                        cy = (box[1] + box[3]) / 2.0
                        self.track_seen[tid] = self.frame_no
                        for li, line in enumerate(ls):
                            cross = line_side(cx, cy, line)
                            if abs(cross) < 5.0:
                                continue
                            side = 1 if cross > 0 else -1
                            key = (tid, li)
                            pside = self.track_side.get(key)
                            if pside is not None and pside != side:
                                with lock:
                                    sources[self.name]["counters"][li] = \
                                        sources[self.name]["counters"].get(li, 0) + 1
                            self.track_side[key] = side

            annotated = r.plot() if r.boxes is not None else frame
            draw_zones(annotated, zs)

            with lock:
                cs = dict(sources[self.name]["counters"])
            draw_lines(annotated, ls, cs)

            counts = {}
            if r.boxes is not None:
                for box in r.boxes:
                    nm = self.model.names[int(box.cls[0])]
                    counts[nm] = counts.get(nm, 0) + 1

            postures = {}
            if self.model_task == "pose" and getattr(r, "keypoints", None) is not None:
                try:
                    kp_xy = r.keypoints.xy.cpu().numpy()
                    kp_conf = r.keypoints.conf.cpu().numpy()
                    for xy, cf in zip(kp_xy, kp_conf):
                        kp = np.hstack([xy, cf[:, None]])
                        p = classify_posture(kp)
                        postures[p] = postures.get(p, 0) + 1
                except Exception:
                    pass

            now = time.time()
            fps = 1.0 / max(now - prev, 1e-6)
            prev = now
            if self.frame_no % 100 == 0:
                print(f"[{self.name}] FPS={fps:.1f} boxes={sum(counts.values())}", flush=True)

            cv2.putText(annotated, f"{self.name}  [{self.model_name}]", (20, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(annotated, f"IN-ZONE: {sum(counts.values())}", (20, 64),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
            cv2.putText(annotated, f"FPS: {fps:.1f}  Zones: {len(zs)}  Lines: {len(ls)}  Total Cross: {sum(cs.values())}",
                        (20, 114), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            y = 174
            for nm, n in sorted(counts.items(), key=lambda kv: -kv[1]):
                cv2.putText(annotated, f"{nm}: {n}", (20, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                y += 40
            for p in ("standing", "lying", "falling", "unknown"):
                if postures.get(p):
                    color = (0, 255, 0) if p == "standing" else (0, 0, 255) if p == "lying" else (0, 200, 255)
                    cv2.putText(annotated, f"{p}: {postures[p]}", (20, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    y += 40

            stale = [k for k, fn in self.track_seen.items() if self.frame_no - fn > 60]
            for k in stale:
                self.track_seen.pop(k, None)
                for li in range(len(ls)):
                    self.track_side.pop((k, li), None)

            ok, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ok:
                with lock:
                    self.latest_jpeg = jpeg.tobytes()

            self.push_async(annotated)


def start_worker(name):
    with lock:
        if name in workers and workers[name].is_alive():
            return
        w = SourceWorker(name)
        workers[name] = w
    w.start()


def stop_worker(name):
    with lock:
        w = workers.pop(name, None)
    if w:
        w.stop()


INDEX_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Multi-Source Detection</title>
<style>
  body { background:#111; color:#eee; font-family: system-ui; padding:30px; }
  h1 { font-size:22px; }
  table { border-collapse:collapse; width:100%; max-width:900px; margin-top:20px; }
  th,td { border:1px solid #444; padding:10px 14px; text-align:left; }
  th { background:#222; }
  .add { display:flex; gap:8px; margin-top:24px; max-width:900px; }
  input { flex:1; background:#222; color:#eee; border:1px solid #555; border-radius:6px; padding:10px; font-size:14px; }
  select { background:#222; color:#eee; border:1px solid #555; border-radius:6px; padding:6px; font-size:13px; }
  button { background:#0a7; color:#fff; border:none; border-radius:6px; padding:10px 18px; cursor:pointer; font-size:14px; }
  button.del { background:#a33; }
  button.view { background:#2a2a2a; border:1px solid #555; }
  button.preview { background:#2a5; border:1px solid #555; }
  button.pause { background:#c80; border:1px solid #555; }
  button.play { background:#0a7; border:1px solid #555; }
  button.stopall { background:#a33; border:none; }
  button.startall { background:#0a7; border:none; }
  code { background:#000; padding:2px 6px; border-radius:4px; }
  .hdr { display:flex; align-items:center; gap:12px; flex-wrap:wrap; }
  .tvwall-btn { background:#0a7; color:#fff; border:none; border-radius:6px; padding:10px 18px; cursor:pointer; font-size:15px; }
  .dot { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:4px; }
  .dot.on { background:#0f0; }
  .dot.off { background:#666; }
</style>
</head>
<body>
<div class="hdr">
  <h1>多來源偵測管理</h1>
  <button class="tvwall-btn" onclick="location.href='/tvwall'">📺 TV Wall</button>
  <button class="startall" onclick="allStart()">▶ 啟動全部</button>
  <button class="stopall" onclick="allStop()">⏹ 停止全部</button>
</div>
<table id="tbl">
  <tr><th>名稱</th><th>來源 URL</th><th>模型</th><th>RTSP 輸出</th><th>操作</th></tr>
</table>
<div class="add">
  <input id="name" placeholder="來源名稱 (英文/數字)">
  <input id="url" placeholder="URL: rtsp://... 或 http://....m3u8 / .flv / .mp4 / .mjpeg">
  <button onclick="addSource()">＋ 加入來源</button>
</div>
<div class="add" style="margin-top:10px">
  <input id="modelFile" type="file" accept=".pt">
  <button onclick="uploadModel()">上傳模型 (.pt)</button>
</div>
<script>
let MODELS = {};
async function load() {
  const [r1, r2] = await Promise.all([fetch('/api/sources'), fetch('/api/models')]);
  const data = await r1.json();
  MODELS = (await r2.json()).models || {};
  const tbl = document.getElementById('tbl');
  tbl.innerHTML = '<tr><th>名稱</th><th>來源 URL</th><th>模型</th><th>RTSP 輸出</th><th>操作</th></tr>';
  for (const [name, cfg] of Object.entries(data.sources)) {
    const tr = document.createElement('tr');
    const opts = Object.keys(MODELS).map(m =>
      `<option value="${m}" ${m === cfg.model ? 'selected' : ''}>${MODELS[m].label}</option>`).join('');
    const running = cfg.running;
    const toggle = running
      ? `<button class="pause" onclick="toggleSource('${name}','stop')">⏹ 停止</button>`
      : `<button class="play" onclick="toggleSource('${name}','start')">▶ 啟動</button>`;
    tr.innerHTML = `<td><span class="dot ${running ? 'on' : 'off'}"></span>${name}</td>` +
      `<td><code>${cfg.url}</code></td>` +
      `<td><select onchange="setModel('${name}', this.value)">${opts}</select></td>` +
      `<td><code>rtsp://${location.hostname}:8554/${name}</code></td>` +
      `<td><button class="preview" onclick="openPreview('${name}')">預覽</button> ` +
      `<button class="view" onclick="location.href='/view/${name}'">設定區域/紅線</button> ` +
      `${toggle} ` +
      `<button class="del" onclick="delSource('${name}')">移除</button></td>`;
    tbl.appendChild(tr);
  }
}
async function toggleSource(name, action) {
  await fetch('/api/sources/' + name + '/' + action, { method:'POST' });
  load();
}
async function allStart() {
  await fetch('/api/start', { method:'POST' });
  load();
}
async function allStop() {
  await fetch('/api/stop', { method:'POST' });
  load();
}
function openPreview(name) {
  window.open('/preview/' + name, 'preview_' + name,
    'width=880,height=520,resizable=yes,scrollbars=no');
}
async function uploadModel() {
  const f = document.getElementById('modelFile').files[0];
  if (!f) { alert('請選取 .pt 檔案'); return; }
  await fetch('/api/models/upload?name=' + encodeURIComponent(f.name), { method:'POST', body: f });
  document.getElementById('modelFile').value = '';
  load();
}
async function setModel(name, model) {
  await fetch('/api/sources/' + name + '/model', { method:'POST',
    headers:{'Content-Type':'application/json'}, body: JSON.stringify({ model }) });
}
async function addSource() {
  const name = document.getElementById('name').value.trim();
  const url = document.getElementById('url').value.trim();
  if (!name || !url) { alert('請填名稱與 URL'); return; }
  await fetch('/api/sources', { method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ name, url }) });
  load();
}
async function delSource(name) {
  if (!confirm('確定移除 ' + name + ' ?')) return;
  await fetch('/api/sources/' + name, { method:'DELETE' });
  load();
}
load();
</script>
</body>
</html>
"""

VIEW_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>__NAME__ - Detection Zones</title>
<style>
  body { margin:0; background:#111; color:#eee; font-family: system-ui; }
  #toolbar { position:fixed; top:12px; left:12px; z-index:20; display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
  button { background:#2a2a2a; color:#eee; border:1px solid #555; border-radius:6px; padding:8px 14px; font-size:14px; cursor:pointer; }
  button:hover { background:#3a3a3a; }
  button.active { background:#0a7; border-color:#0a7; color:#fff; }
  #drawZoneBtn.active { background:#0a7; }
  #drawLineBtn.active { background:#c00; }
  #status { font-size:13px; color:#9c9; }
  #wrap { position:relative; display:inline-block; margin-top:52px; }
  #video { display:block; max-width:100vw; }
  #canvas { position:absolute; top:0; left:0; }
  #classesBar { position:fixed; top:58px; left:12px; z-index:20; display:none; gap:8px; align-items:center; flex-wrap:wrap;
    background:#1a1a1a; border:1px solid #444; border-radius:6px; padding:6px 10px; max-width:70vw; }
  #classesBar.show { display:flex; }
  #classesBar label { font-size:12px; display:inline-flex; align-items:center; gap:3px; }
</style>
</head>
<body>
<div id="toolbar">
  <a href="/" style="color:#9cf">← 回列表</a>
  <select id="modelSel" onchange="setModel(this.value)"></select>
  <select id="imgszSel" onchange="setImgsz(this.value)">
    <option value="384">384 (最快)</option>
    <option value="480">480 (快)</option>
    <option value="640">640</option>
    <option value="768">768</option>
    <option value="960">960</option>
    <option value="1280">1280 (準)</option>
  </select>
  <button id="toggleClassesBtn">類別選擇</button>
  <button id="drawZoneBtn">＋ 偵測區域</button>
  <button id="drawLineBtn">＋ 紅線(計數)</button>
  <button id="finishBtn" disabled>完成區域</button>
  <button id="saveBtn">儲存</button>
  <button id="resetBtn">重置計數</button>
  <button id="clearBtn">清除全部</button>
  <span id="status">__NAME__ 載入中…</span>
</div>
<div id="classesBar"><span>類別:</span><span id="classBoxes"></span><button id="saveClassesBtn" style="padding:4px 8px;font-size:12px">套用類別</button></div>
<div id="wrap">
  <img id="video" src="/stream/__NAME__" alt="stream">
  <canvas id="canvas"></canvas>
</div>
<script>
const NAME = "__NAME__";
const img = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const drawZoneBtn = document.getElementById('drawZoneBtn');
const drawLineBtn = document.getElementById('drawLineBtn');
const finishBtn = document.getElementById('finishBtn');
const saveBtn = document.getElementById('saveBtn');
const resetBtn = document.getElementById('resetBtn');
const clearBtn = document.getElementById('clearBtn');
const statusEl = document.getElementById('status');
const toggleClassesBtn = document.getElementById('toggleClassesBtn');
const classesBar = document.getElementById('classesBar');

toggleClassesBtn.addEventListener('click', () => {
  classesBar.classList.toggle('show');
});

let zones = [];
let lines = [];
let current = [];
let mode = null;

function resize() { canvas.width = img.clientWidth; canvas.height = img.clientHeight; redraw(); }
window.addEventListener('resize', resize);
img.addEventListener('load', () => { resize(); loadConfig(); });

function toFrame(ev) {
  const sx = img.naturalWidth / img.clientWidth;
  const sy = img.naturalHeight / img.clientHeight;
  const rect = canvas.getBoundingClientRect();
  return [(ev.clientX - rect.left) * sx, (ev.clientY - rect.top) * sy];
}

function redraw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const sx = canvas.width / img.naturalWidth;
  const sy = canvas.height / img.naturalHeight;
  zones.forEach((poly, i) => drawPoly(poly, '#00ffaa', 'Z' + (i + 1)));
  lines.forEach((l, i) => drawLine(l, 'L' + (i + 1)));
  if (current.length) {
    if (mode === 'zone') {
      ctx.beginPath();
      current.forEach((p, i) => { const x=p[0]*sx, y=p[1]*sy; i===0?ctx.moveTo(x,y):ctx.lineTo(x,y); });
      ctx.strokeStyle = '#ffd400'; ctx.lineWidth = 2; ctx.stroke();
      current.forEach(p => { ctx.beginPath(); ctx.arc(p[0]*sx,p[1]*sy,4,0,2*Math.PI); ctx.fillStyle='#ffd400'; ctx.fill(); });
    } else if (mode === 'line') {
      const a = current[0];
      ctx.beginPath(); ctx.arc(a[0]*sx,a[1]*sy,5,0,2*Math.PI); ctx.fillStyle='#ffd400'; ctx.fill();
      if (current.length === 2) {
        ctx.beginPath(); ctx.moveTo(a[0]*sx,a[1]*sy); ctx.lineTo(current[1][0]*sx,current[1][1]*sy);
        ctx.strokeStyle='#ff0000'; ctx.lineWidth=3; ctx.stroke();
      }
    }
  }
}

function drawPoly(poly, color, label) {
  const sx = canvas.width / img.naturalWidth, sy = canvas.height / img.naturalHeight;
  ctx.beginPath();
  poly.forEach((p, i) => { const x=p[0]*sx, y=p[1]*sy; i===0?ctx.moveTo(x,y):ctx.lineTo(x,y); });
  ctx.closePath(); ctx.fillStyle = color + '33'; ctx.fill();
  ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.stroke();
  if (label) {
    const cx = poly.reduce((s,p)=>s+p[0],0)/poly.length*sx;
    const cy = poly.reduce((s,p)=>s+p[1],0)/poly.length*sy;
    ctx.fillStyle = color; ctx.font='bold 16px system-ui'; ctx.fillText(label, cx-6, cy+6);
  }
}

function drawLine(l, label) {
  const sx = canvas.width / img.naturalWidth, sy = canvas.height / img.naturalHeight;
  ctx.beginPath(); ctx.moveTo(l[0][0]*sx,l[0][1]*sy); ctx.lineTo(l[1][0]*sx,l[1][1]*sy);
  ctx.strokeStyle = '#ff0000'; ctx.lineWidth = 3; ctx.stroke();
  const mx=(l[0][0]+l[1][0])/2*sx, my=(l[0][1]+l[1][1])/2*sy;
  ctx.fillStyle='#ff0000'; ctx.font='bold 16px system-ui'; ctx.fillText(label, mx-14, my-8);
}

canvas.addEventListener('click', (ev) => {
  if (!mode) return;
  const p = toFrame(ev);
  if (mode === 'zone') { current.push(p); finishBtn.disabled = current.length < 3; }
  else if (mode === 'line') {
    current.push(p);
    if (current.length === 2) { lines.push([current[0], current[1]]); current = []; statusEl.textContent='紅線已加入（未儲存）'; }
  }
  redraw();
});
canvas.addEventListener('dblclick', (ev) => { if (mode === 'zone') { ev.preventDefault(); finishPoly(); } });

function setMode(m) {
  mode = m; current = []; finishBtn.disabled = true;
  drawZoneBtn.classList.toggle('active', m === 'zone');
  drawLineBtn.classList.toggle('active', m === 'line');
  canvas.style.cursor = m ? 'crosshair' : 'default';
  redraw();
}
drawZoneBtn.addEventListener('click', () => setMode(mode === 'zone' ? null : 'zone'));
drawLineBtn.addEventListener('click', () => setMode(mode === 'line' ? null : 'line'));
finishBtn.addEventListener('click', finishPoly);
function finishPoly() {
  if (current.length >= 3) { zones.push(current.slice()); statusEl.textContent='區域已加入（未儲存）'; }
  current = []; finishBtn.disabled = true; redraw();
}

const cfgUrl = '/api/sources/' + NAME + '/config';
saveBtn.addEventListener('click', () => {
  fetch(cfgUrl, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({zones, lines}) })
    .then(r => statusEl.textContent = r.ok ? `已儲存 ${zones.length} 區域 / ${lines.length} 紅線` : '儲存失敗');
});
resetBtn.addEventListener('click', () => {
  fetch('/api/sources/' + NAME + '/reset', { method:'POST' }).then(() => statusEl.textContent='計數已重置');
});
clearBtn.addEventListener('click', () => {
  zones = []; lines = []; current = []; setMode(null);
  fetch(cfgUrl, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({zones:[], lines:[]}) })
    .then(() => statusEl.textContent='已清除');
});
function loadConfig() {
  fetch(cfgUrl).then(r => r.json()).then(d => {
    zones = d.zones || []; lines = d.lines || [];
    if (d.imgsz) document.getElementById('imgszSel').value = String(d.imgsz);
    statusEl.textContent = '就緒'; redraw();
  });
}
async function setImgsz(v) {
  await fetch('/api/sources/' + NAME + '/config', { method:'POST',
    headers:{'Content-Type':'application/json'}, body: JSON.stringify({ zones, lines, imgsz: parseInt(v) }) });
  statusEl.textContent = '已設定 imgsz=' + v;
}
async function setModel(model) {
  await fetch('/api/sources/' + NAME + '/model', { method:'POST',
    headers:{'Content-Type':'application/json'}, body: JSON.stringify({ model }) });
  statusEl.textContent = '已切換模型: ' + model;
  loadClasses(model);
}
async function loadClasses(model) {
  const box = document.getElementById('classBoxes');
  box.innerHTML = '載入中…';
  const r = await fetch('/api/models/' + model + '/classes');
  const d = await r.json();
  const names = d.names || {};
  const cfg = await (await fetch('/api/sources/' + NAME + '/config')).json();
  const selected = cfg.classes;  // list or "default" or null
  const sel = (selected === 'default') ? null : selected; // null => all
  const allChecked = (sel === null);
  box.innerHTML = Object.entries(names).map(([i, n]) =>
    `<label><input type="checkbox" value="${i}" ${allChecked ? 'checked' : (sel || []).includes(parseInt(i)) ? 'checked' : ''}>${n}</label>`
  ).join('');
  document.getElementById('saveClassesBtn').onclick = () => saveClasses();
}
function saveClasses() {
  const checked = [...document.querySelectorAll('#classBoxes input:checked')].map(c => parseInt(c.value));
  const boxes = [...document.querySelectorAll('#classBoxes input')];
  const classes = (checked.length === boxes.length || checked.length === 0) ? null : checked;
  fetch('/api/sources/' + NAME + '/config', { method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ zones, lines, classes }) })
    .then(() => statusEl.textContent = '類別已套用');
}
fetch('/api/models').then(r => r.json()).then(d => {
  const sel = document.getElementById('modelSel');
  const models = d.models || {};
  Object.keys(models).forEach(m => {
    const o = document.createElement('option');
    o.value = m; o.textContent = models[m].label;
    sel.appendChild(o);
  });
});
fetch('/api/sources').then(r => r.json()).then(d => {
  const cur = d.sources[NAME] && d.sources[NAME].model;
  if (cur) { document.getElementById('modelSel').value = cur; loadClasses(cur); }
});
 </script>
 </body>
 </html>
 """


PREVIEW_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>__NAME__ - Preview</title>
<style>
  body { margin:0; background:#000; display:flex; flex-direction:column; align-items:center; }
  .top { color:#eee; font-family:system-ui; padding:8px 16px; display:flex; gap:12px; align-items:center; }
  .top a { color:#9cf; text-decoration:none; }
  img { max-width:100vw; max-height:calc(100vh - 40px); }
</style>
</head>
<body>
<div class="top"><a href="/">← 回列表</a><span>__NAME__ 即時預覽</span></div>
<img src="/stream/__NAME__" alt="stream">
</body>
</html>
"""


TVWALL_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>TV Wall</title>
<style>
  html, body { margin:0; height:100%; background:#000; overflow:hidden; }
  .bar { position:fixed; top:10px; left:10px; z-index:10; display:flex; gap:8px; }
  .bar a, .bar button { background:#2a2a2a; color:#eee; border:1px solid #555; border-radius:6px;
    padding:6px 12px; font-size:13px; text-decoration:none; cursor:pointer; }
  #grid { display:grid; gap:3px; padding:3px; width:100vw; height:100vh; box-sizing:border-box; }
  .cell { position:relative; background:#111; overflow:hidden; cursor:pointer; min-width:0; min-height:0; }
  .cell img { width:100%; height:100%; object-fit:contain; }
  .cell .label { position:absolute; top:6px; left:6px; color:#0f0; font:bold 13px system-ui;
    background:rgba(0,0,0,0.55); padding:2px 8px; border-radius:4px; }
</style>
</head>
<body>
<div class="bar"><a href="/">← 回列表</a><button onclick="reload()">重新整理</button><span id="count" style="color:#9c9;font-size:13px"></span></div>
<div id="grid"></div>
<script>
function reload() {
  fetch('/api/sources').then(r => r.json()).then(d => {
    const grid = document.getElementById('grid');
    grid.innerHTML = '';
    const names = Object.keys(d.sources);
    document.getElementById('count').textContent = names.length + ' 路';
    const n = names.length || 1;
    const cols = Math.ceil(Math.sqrt(n));
    const rows = Math.ceil(n / cols);
    grid.style.gridTemplateColumns = `repeat(${cols}, 1fr)`;
    grid.style.gridTemplateRows = `repeat(${rows}, 1fr)`;
    for (const name of names) {
      const cell = document.createElement('div');
      cell.className = 'cell';
      cell.innerHTML = `<img src="/stream/${name}"><div class="label">${name}</div>`;
      cell.onclick = () => location.href = '/view/' + name;
      grid.appendChild(cell);
    }
  });
}
reload();
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def _json(self, obj, status=200):
        data = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = unquote(self.path.split("?")[0]).strip("/")
        parts = path.split("/") if path else []

        if not parts:
            html = INDEX_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)
            return

        if parts[0] == "view" and len(parts) == 2:
            name = parts[1]
            html = VIEW_HTML.replace("__NAME__", name).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)
            return

        if parts[0] == "preview" and len(parts) == 2:
            name = parts[1]
            html = PREVIEW_HTML.replace("__NAME__", name).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)
            return

        if parts[0] == "tvwall":
            html = TVWALL_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)
            return

        if parts[0] == "stream" and len(parts) == 2:
            name = parts[1]
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            last_sent = None
            try:
                self.connection.settimeout(1.0)
                while True:
                    r, _, _ = select.select([self.connection], [], [], 0)
                    if r:
                        try:
                            if self.connection.recv(1, socket.MSG_PEEK) == b"":
                                break
                        except (OSError, ConnectionError):
                            break
                    with lock:
                        w = workers.get(name)
                        jpeg = w.latest_jpeg if w else None
                    if jpeg is None:
                        jpeg = placeholder_jpeg
                    if jpeg != last_sent:
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                        self.wfile.flush()
                        last_sent = jpeg
                    time.sleep(0.03 if jpeg is not placeholder_jpeg else 0.3)
            except (BrokenPipeError, ConnectionResetError, socket.timeout, OSError):
                pass
            finally:
                try:
                    self.connection.close()
                except OSError:
                    pass
            return

        if parts[0] == "api" and parts[1] == "sources" and len(parts) == 2:
            with lock:
                data = {k: {"url": v["url"], "model": v.get("model", MODEL_NAME),
                            "classes": v.get("classes", "default"),
                            "enabled": v.get("enabled", True),
                            "running": k in workers and workers[k].is_alive()}
                        for k, v in sources.items()}
            self._json({"sources": data})
            return

        if parts[0] == "api" and parts[1] == "models" and len(parts) == 2:
            self._json({"models": MODEL_CATALOG})
            return

        if parts[0] == "api" and parts[1] == "models" and len(parts) == 4 and parts[3] == "classes":
            name = parts[2]
            names, task = get_model_info(name)
            self._json({"names": names, "task": task})
            return

        if parts[0] == "api" and parts[1] == "sources" and len(parts) == 4 and parts[3] == "config":
            name = parts[2]
            with lock:
                cfg = sources.get(name)
            if not cfg:
                self.send_error(404)
                return
            self._json({"zones": cfg["zones"], "lines": cfg["lines"],
                        "classes": cfg.get("classes", "default"),
                        "imgsz": cfg.get("imgsz", IMGSZ)})
            return

        self.send_error(404)

    def do_POST(self):
        path = unquote(self.path.split("?")[0]).strip("/")
        parts = path.split("/") if path else []

        if parts[0] == "api" and parts[1] == "sources" and len(parts) == 2:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            name = body.get("name", "").strip()
            url = body.get("url", "").strip()
            model = body.get("model", MODEL_NAME)
            if not name or not url:
                self.send_error(400)
                return
            with lock:
                sources[name] = {"url": url, "model": model, "classes": "default",
                                 "imgsz": IMGSZ, "enabled": True,
                                 "zones": [], "lines": [], "counters": {}}
                save_persisted()
            start_worker(name)
            self._json({"ok": True})
            return

        if parts[0] == "api" and parts[1] == "models" and parts[2] == "upload" and len(parts) == 3:
            qs = parse_qs(urlparse(self.path).query)
            fname = (qs.get("name", [""])[0]).strip()
            if not fname.endswith(".pt"):
                self.send_error(400)
                return
            fname = os.path.basename(fname)  # prevent path traversal
            length = int(self.headers.get("Content-Length", 0))
            data = self.rfile.read(length)
            with open(os.path.join(MODELS_DIR, fname), "wb") as f:
                f.write(data)
            # validate + register
            names, task = get_model_info(fname)
            MODEL_CATALOG[fname] = {"label": f"自訂: {fname}", "classes": None, "task": task}
            self._json({"ok": True, "names": names})
            return

        if parts[0] == "api" and parts[1] == "sources" and len(parts) == 4 and parts[3] == "model":
            name = parts[2]
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            model = body.get("model", "").strip()
            if not model:
                self.send_error(400)
                return
            with lock:
                if name not in sources:
                    self.send_error(404)
                    return
                sources[name]["model"] = model
                sources[name]["classes"] = "default"
                save_persisted()
            self._json({"ok": True})
            return

        if parts[0] == "api" and parts[1] == "sources" and len(parts) == 4 and parts[3] == "config":
            name = parts[2]
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            with lock:
                if name not in sources:
                    self.send_error(404)
                    return
                sources[name]["zones"] = [[[float(p[0]), float(p[1])] for p in poly] for poly in body.get("zones", [])]
                sources[name]["lines"] = [[[float(l[0][0]), float(l[0][1])], [float(l[1][0]), float(l[1][1])]]
                                          for l in body.get("lines", [])]
                if "classes" in body:
                    sources[name]["classes"] = body["classes"]
                if "imgsz" in body:
                    sources[name]["imgsz"] = int(body["imgsz"])
                save_persisted()
            self._json({"ok": True})
            return

        if parts[0] == "api" and parts[1] == "sources" and len(parts) == 4 and parts[3] == "reset":
            name = parts[2]
            with lock:
                if name in sources:
                    sources[name]["counters"] = {}
                    save_persisted()
            self._json({"ok": True})
            return

        if parts[0] == "api" and parts[1] == "sources" and len(parts) == 4 and parts[3] == "start":
            name = parts[2]
            with lock:
                if name not in sources:
                    self.send_error(404)
                    return
                sources[name]["enabled"] = True
                save_persisted()
            start_worker(name)
            self._json({"ok": True})
            return

        if parts[0] == "api" and parts[1] == "sources" and len(parts) == 4 and parts[3] == "stop":
            name = parts[2]
            with lock:
                if name not in sources:
                    self.send_error(404)
                    return
                sources[name]["enabled"] = False
                save_persisted()
            stop_worker(name)
            self._json({"ok": True})
            return

        if parts[0] == "api" and len(parts) == 2 and parts[1] == "start":
            with lock:
                names = list(sources.keys())
                for n in names:
                    sources[n]["enabled"] = True
                save_persisted()
            for n in names:
                start_worker(n)
            self._json({"ok": True, "started": len(names)})
            return

        if parts[0] == "api" and len(parts) == 2 and parts[1] == "stop":
            with lock:
                names = list(sources.keys())
                for n in names:
                    sources[n]["enabled"] = False
                save_persisted()
            for n in names:
                stop_worker(n)
            self._json({"ok": True, "stopped": len(names)})
            return

        self.send_error(404)

    def do_DELETE(self):
        path = unquote(self.path.split("?")[0]).strip("/")
        parts = path.split("/") if path else []
        if parts[0] == "api" and parts[1] == "sources" and len(parts) == 3:
            name = parts[2]
            stop_worker(name)
            with lock:
                sources.pop(name, None)
                save_persisted()
            self._json({"ok": True})
            return
        self.send_error(404)

    def log_message(self, *args):
        pass


def main():
    load_persisted()
    scan_models_dir()
    with lock:
        names = [n for n, c in sources.items() if c.get("enabled", True)]
    for name in names:
        start_worker(name)
        print(f"started source: {name}", flush=True)

    server = ThreadingHTTPServer(("0.0.0.0", HTTP_PORT), Handler)
    print(f"Open http://localhost:{HTTP_PORT} in your browser", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
