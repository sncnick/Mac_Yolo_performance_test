import sys
import time

import cv2
from ultralytics import YOLO

DEFAULT_URL = "https://g2r.projectsndev.com/api/stream.mp4?src=autoid_PO_11_TEST"

VIOLATION = {6, 7, 8}  # NO-Hardhat, NO-Mask, NO-Safety Vest
SAFE = {4, 5, 12}      # Hardhat, Mask, Safety Vest
PERSON = 9


def main():
    url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URL
    model = YOLO("construction.pt")

    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        raise SystemExit(f"無法開啟串流: {url}")

    prev = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        results = model.predict(
            source=frame,
            conf=0.35,
            imgsz=960,
            device="mps",
            verbose=False,
        )

        annotated = results[0].plot()

        counts = {}
        if results[0].boxes is not None:
            for box in results[0].boxes:
                name = model.names[int(box.cls[0])]
                counts[name] = counts.get(name, 0) + 1

        people = counts.get("Person", 0)
        violations = sum(counts.get(model.names[c], 0) for c in VIOLATION)

        if violations > 0:
            status, color = f"VIOLATION x{violations}", (0, 0, 255)
        elif people > 0:
            status, color = f"SAFE ({people} people)", (0, 255, 0)
        else:
            status, color = "NO PERSON", (150, 150, 150)

        now = time.time()
        fps = 1.0 / max(now - prev, 1e-6)
        prev = now

        cv2.putText(annotated, status, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4)
        cv2.putText(annotated, f"FPS: {fps:.1f}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        y = 180
        for name in ("Person", "Hardhat", "NO-Hardhat", "Safety Vest", "NO-Safety Vest", "Mask", "NO-Mask"):
            n = counts.get(name, 0)
            if n:
                c = (0, 0, 255) if name.startswith("NO-") else (255, 255, 255)
                cv2.putText(annotated, f"{name}: {n}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 1, c, 2)
                y += 40

        cv2.imshow("Construction Site Detection", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
