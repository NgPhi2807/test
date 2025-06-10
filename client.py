import cv2
import os
from PIL import Image
from facenet_pytorch import MTCNN
import torch
import requests
from io import BytesIO
import sys
import time
import numpy as np

student_id = 11
subject_date_id = input("Nh?p subject_date_id: ").strip()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(keep_all=False, device=device)

NUM_IMAGES = 100
cap = cv2.VideoCapture(0)
i = 0
mode = 'idle'

RECOGNITION_URL = "http://127.0.0.1:8000/api/recognize/"  # ?? Thay b?ng IP server th?t n?u c?n

print('[INFO] Press "T" to start collecting images or "A" to start recognition')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)

    boxes, _ = mtcnn.detect(img)
    if boxes is not None and len(boxes) > 0:
        x1, y1, x2, y2 = [int(v) for v in boxes[0]]

        if mode == 'collect' and i < NUM_IMAGES:
            face = img.crop((x1, y1, x2, y2)).resize((160, 160))
            buffer = BytesIO()
            face.save(buffer, format="JPEG")
            buffer.seek(0)

            response = requests.post(
                "http://127.0.0.1:8000/api/upload-face/",
                files={"image": (f"{i}.jpg", buffer, "image/jpeg")},
                data={
                    "student_id": student_id,
                    }
            )

            if response.status_code == 200:
                print(f"[{i}] Uploaded")
                i += 1
            else:
                print(f"[{i}] Upload failed: {response.text}")

    if mode == 'recognize':
        _, img_encoded = cv2.imencode('.jpg', frame)
        files = {'image': ('frame.jpg', img_encoded.tobytes(), 'image/jpeg')}
        try:
            res = requests.post(RECOGNITION_URL, files=files, data={"subject_date_id": subject_date_id}, timeout=10)
            if res.status_code == 200:
                result = res.json()
                boxes = result.get("boxes", [])
                names = result.get("names", [])
                scores = result.get("scores", [])
                if names == "NF":
                    height, width, _ = frame.shape
                    text = "SubjectDate not found"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1
                    thickness = 2
                    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
                    text_x = (width - text_size[0]) // 2
                    text_y = (height + text_size[1]) // 2

                    cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)
                else:
                    for (box, name, score) in zip(boxes, names, scores):
                        x1, y1, x2, y2 = map(int, box)
                        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f'{name} {score:.2f}', (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            else:
                print(f"[ERROR] Server error: {res.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Request failed: {e}")

    cv2.imshow('Face Client', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('t'):
        mode = 'collect'
        i = 0
        print("[INFO] Collect mode activated")
    elif key == ord('a'):
        mode = 'recognize'
        print("[INFO] Recognize mode activated")

cap.release()
cv2.destroyAllWindows()
