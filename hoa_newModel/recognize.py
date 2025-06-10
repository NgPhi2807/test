import cv2
import numpy as np
import mtcnn
import os
import pickle
import time
from architecture_embedding import InceptionResNetV2
from scipy.spatial.distance import cosine
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
from sklearn.preprocessing import Normalizer

confidence_t = 0.8
recognition_t = 0.4
required_size = (160, 160)
encodings_path = r'F:\codePython\PBL5\face_recognize\hoa_newModel\encodings1.pkl'
model_path = r'F:\codePython\PBL5\face_recognize\hoa_newModel\embedding_model_new6.h5'

# Khởi tạo mô hình, detector, normalizer
l2_normalizer = Normalizer('l2')
face_encoder = InceptionResNetV2()
face_encoder.load_weights(model_path)
face_detector = mtcnn.MTCNN()

def load_pickle(path):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return {}

def extract_face(img, box):
    x1, y1, w, h = box
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + w, y1 + h
    face = img[y1:y2, x1:x2]
    return face

def get_embedding(face):
    try:
        face = cv2.resize(face, required_size)
    except:
        return None
    face = preprocess_input(face.astype(np.float32))
    encode = face_encoder.predict(np.expand_dims(face, axis=0), verbose=0)[0]
    return l2_normalizer.transform(encode.reshape(1, -1))[0]

def recognize():
    encoding_dict = load_pickle(encodings_path)
    if not encoding_dict:
        print("[WARNING] Chưa có dữ liệu embedding để nhận diện.")
        return

    cap = cv2.VideoCapture(0)
    print("[INFO] Nhấn 'q' để thoát.")
    last_results = []
    frame_count = 0
    process_interval = 2  # Cứ 2 frame dò mặt 1 lần để tăng tốc

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = []

        if frame_count % process_interval == 0:
            results = face_detector.detect_faces(img_rgb)
            last_results = results
            print(f"[DEBUG] Detected {len(results)} faces")
        else:
            results = last_results

        for res in results:
            if res['confidence'] < confidence_t:
                continue
            face = extract_face(img_rgb, res['box'])
            embedding = get_embedding(face)
            name = "unknown"
            min_dist = float('inf')

            if embedding is not None:
                for db_name, db_encodes in encoding_dict.items():
                    for db_encode in db_encodes:
                        dist = cosine(db_encode, embedding)
                        if dist < recognition_t and dist < min_dist:
                            name = db_name
                            min_dist = dist

            x1, y1, w, h = res['box']
            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1 + w, y1 + h
            label = f"{name}" if name == "unknown" else f"{name} ({min_dist:.2f})"
            color = (0, 0, 255) if name == "unknown" else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize()
