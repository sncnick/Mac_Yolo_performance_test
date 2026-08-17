import select
import subprocess
import sys
import time

import cv2
import numpy as np
from ultralytics import YOLO

DEFAULT_URL = "rtsp://admin:autoid3568@@autoidpf.bdcode.com:13572/Streaming/Channels/101"

VEHICLE_CLASSES = [1, 2, 3, 5, 7]  # bicycle, car, motorcycle, bus, truck

W, H = 1280, 720
FRAME_SIZE = W * H * 3


def start_ffmpeg(url):
    cmd = [
        "ffmpeg", "-v", "error",
        "-fflags", "nobuffer", "-flags", "low_delay",
        "-probesize", "32", "-analyzeduration", "0",
        "-rtsp_transport", "tcp", "-stimeout", "5000000",
        "-i", url,
        "-vf", "scale=1280:720,setsar=1",
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
        if n is None or n == 0:
            return None
        filled += n
    return np.frombuffer(buf, dtype=np.uint8).reshape(H, W, 3)


def main():
    url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URL
    model = YOLO("yolo11s.pt")

    proc = start_ffmpeg(url)
    prev = time.time()

    while True:
        frame = read_frame(proc)
        if frame is None:
            proc.kill()
            proc = start_ffmpeg(url)
            continue

        results = model.predict(
            source=frame,
            classes=VEHICLE_CLASSES,
            conf=0.35,
            imgsz=960,
            device="mps",
            verbose=False,
        )

        counts = {}
        if results[0].boxes is not None:
            for box in results[0].boxes:
                name = model.names[int(box.cls[0])]
                counts[name] = counts.get(name, 0) + 1

        annotated = results[0].plot()

        total = sum(counts.values())
        now = time.time()
        fps = 1.0 / max(now - prev, 1e-6)
        prev = now

        cv2.putText(annotated, f"Vehicles: {total}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
        cv2.putText(annotated, f"FPS: {fps:.1f}", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        y = 170
        for name, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            cv2.putText(annotated, f"{name}: {n}", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            y += 40

        cv2.imshow("Vehicle Detection", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    proc.kill()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
