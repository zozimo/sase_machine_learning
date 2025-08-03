from ultralytics import YOLO
import cv2

model = YOLO('yolov8n.pt')  # Nano model for max speed

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("http://192.168.1.4:8080/video")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        label = model.names[cls]
        conf = float(box.conf[0])

        if conf > 0.5:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    cv2.imshow('YOLOv8n Detection', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
