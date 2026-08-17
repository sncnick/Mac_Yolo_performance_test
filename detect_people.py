import cv2
from ultralytics import YOLO


def main():
    model = YOLO("yolo11n.pt")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("無法開啟鏡頭 (index 0)。請檢查相機權限或用 --camera 指定。")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.predict(
            source=frame,
            classes=[0],  # 0 = person
            conf=0.4,
            device="mps",
            verbose=False,
        )

        frame = results[0].plot()

        people = int(results[0].boxes.shape[0]) if results[0].boxes is not None else 0
        cv2.putText(
            frame,
            f"People: {people}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        cv2.imshow("YOLO Person Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
