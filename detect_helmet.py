import cv2
from ultralytics import YOLO


def main():
    model = YOLO("helmet.pt")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("無法開啟鏡頭 (index 0)。請檢查相機權限。")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.predict(
            source=frame,
            conf=0.4,
            device="mps",
            verbose=False,
        )

        frame = results[0].plot()

        has_hat = False
        no_hat = False
        if results[0].boxes is not None:
            for box in results[0].boxes:
                cls = int(box.cls[0])
                if cls == 0:
                    has_hat = True
                elif cls == 1:
                    no_hat = True

        if has_hat and not no_hat:
            status, color = "HELMET ON", (0, 255, 0)
        elif no_hat and not has_hat:
            status, color = "NO HELMET", (0, 0, 255)
        elif has_hat and no_hat:
            status, color = "MIXED", (0, 200, 255)
        else:
            status, color = "NO PERSON", (150, 150, 150)

        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imshow("Helmet Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
