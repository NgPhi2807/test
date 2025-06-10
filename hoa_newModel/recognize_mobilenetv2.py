
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
from sklearn.cluster import KMeans
import imgaug.augmenters as iaa

# ==== THAM SỐ CẤU HÌNH ====
confidence_t = 0.8  # Confidence threshold for face detection
recognition_t = 0.4  # Initial cosine distance threshold
required_size = (160, 160)
encodings_path = 'encodings.pkl'
model_path = 'embedding_model_new6.h5'
# model_path = 'facenet_weights.h5'

dataset_path = 'datasets'
os.makedirs("encodings", exist_ok=True)

# ==== KHỞI TẠO ====
l2_normalizer = Normalizer('l2')
try:
    face_encoder = InceptionResNetV2()
    face_encoder.load_weights(model_path)
    print("[INFO] Model loaded successfully")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    exit(1)
face_detector = mtcnn.MTCNN()
encoding_dict = {}
last_results = []  # Store last detection results

# ==== AUGMENTATION PIPELINE ====
aug = iaa.Sequential([
    iaa.Sometimes(0.5, iaa.Affine(scale=(0.9, 1.1))),  # Scale 90–110%
    iaa.Sometimes(0.5, iaa.Fliplr(0.5)),  # Horizontal flip
    iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.5))),  # Slight blur
    iaa.Sometimes(0.5, iaa.Multiply((0.8, 1.2))),  # Brightness adjustment
    iaa.Sometimes(0.3, iaa.contrast.LinearContrast((0.8, 1.2))),  # Updated contrast adjustment
])

# ==== LOAD/SAVE ENCODING ====
def load_pickle(path):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return {}

def save_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

# ==== XỬ LÝ ẢNH ====
def preprocess_frame(img_rgb):
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    img_gray = cv2.equalizeHist(img_gray)
    img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    return img_rgb

def extract_face(img, box):
    x1, y1, w, h = box
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + w, y1 + h
    face = img[y1:y2, x1:x2]
    return face, (x1, y1), (x2, y2)

def get_embedding(face):
    try:
        face = cv2.resize(face, required_size)
    except Exception as e:
        print(f"[DEBUG] Failed to resize face: {e}")
        return None
    face = preprocess_input(face.astype(np.float32))
    encode = face_encoder.predict(np.expand_dims(face, axis=0), verbose=0)[0]
    return l2_normalizer.transform(encode.reshape(1, -1))[0]

# ==== TUNE RECOGNITION THRESHOLD ====
def tune_threshold(encoding_dict, dataset_path, num_samples=70):
    same_person_dists = []
    diff_person_dists = []
    for name1 in encoding_dict:
        person_path = os.path.join(dataset_path, name1)
        if not os.path.isdir(person_path):
            continue
        images = np.random.choice(os.listdir(person_path), min(num_samples, len(os.listdir(person_path))),
                                  replace=False)
        for img_file in images:
            img_path = os.path.join(person_path, img_file)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = face_detector.detect_faces(img_rgb)
            for res in faces:
                if res['confidence'] < confidence_t:
                    continue
                face, _, _ = extract_face(img_rgb, res['box'])
                embedding = get_embedding(face)
                if embedding is None:
                    continue
                for db_encode in encoding_dict[name1]:
                    same_person_dists.append(cosine(db_encode, embedding))
                for name2 in encoding_dict:
                    if name1 != name2:
                        for db_encode in encoding_dict[name2]:
                            diff_person_dists.append(cosine(db_encode, embedding))

    if not same_person_dists or not diff_person_dists:
        print("[WARNING] Not enough distances computed. Using default threshold.")
        return recognition_t

    thresholds = np.arange(0.1, 1.0, 0.05)
    best_threshold, best_accuracy = recognition_t, 0
    for t in thresholds:
        correct = sum(d < t for d in same_person_dists) + sum(d > t for d in diff_person_dists)
        accuracy = correct / (len(same_person_dists) + len(diff_person_dists))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = t
    print(f"[INFO] Optimal threshold: {best_threshold}, Accuracy: {best_accuracy:.2f}")
    return best_threshold

# ==== TÍNH TOÁN EMBEDDINGS TỪ DATASET ====
def compute_clustered_embeddings(dataset_path, n_clusters=7):
    encoding_dict = {}
    for name in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, name)
        if not os.path.isdir(person_path):
            continue
        embeddings = []
        for img_file in os.listdir(person_path):
            img_path = os.path.join(person_path, img_file)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = face_detector.detect_faces(img_rgb)
            for res in faces:
                if res['confidence'] < confidence_t:
                    continue
                face, _, _ = extract_face(img_rgb, res['box'])
                # Generate augmented faces
                augmented_faces = aug.augment_images([face] * 3)
                for aug_face in [face] + augmented_faces:
                    embedding = get_embedding(aug_face)
                    if embedding is not None:
                        embeddings.append(embedding)
        if embeddings:
            kmeans = KMeans(n_clusters=min(n_clusters, len(embeddings)), random_state=0)
            kmeans.fit(embeddings)
            encoding_dict[name] = kmeans.cluster_centers_
    save_pickle(encoding_dict, encodings_path)
    print(f"[INFO] Saved clustered embeddings to {encodings_path}")
    return encoding_dict

# ==== ĐĂNG KÝ NGƯỜI MỚI ====
def register_new_face(name):
    print(f"[INFO] Thu thập khuôn mặt cho: {name}")
    cap = cv2.VideoCapture(0)
    collected = []
    frame_count = 0
    collect_interval = 5
    max_real_images = 20
    augment_per_image = 4

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[DEBUG] Failed to read frame")
            continue

        frame_count += 1
        img_rgb = preprocess_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        faces = face_detector.detect_faces(img_rgb)

        for res in faces:
            if res['confidence'] < confidence_t:
                continue
            face, pt1, pt2 = extract_face(img_rgb, res['box'])
            if frame_count % collect_interval == 0:
                # Get embedding for original face
                embedding = get_embedding(face)
                if embedding is not None:
                    collected.append(embedding)
                    # Generate augmented faces
                    augmented_faces = aug.augment_images([face] * augment_per_image)
                    for aug_face in augmented_faces:
                        aug_embedding = get_embedding(aug_face)
                        if aug_embedding is not None:
                            collected.append(aug_embedding)
                    print(f"[INFO] Thu thập được {len(collected)} embeddings (real + augmented)")

            cv2.rectangle(frame, pt1, pt2, (0, 255, 0), 2)
            cv2.putText(frame, f"Thu thập {len(collected)} embeddings", (pt1[0], pt1[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Register Face", frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or len(collected) >= (max_real_images * (1 + augment_per_image)):
            break

    cap.release()
    cv2.destroyAllWindows()

    if collected:
        avg_embedding = np.mean(collected, axis=0)
        encoding_dict[name] = [avg_embedding]
        save_pickle(encoding_dict, encodings_path)
        print(f"[SUCCESS] Đã lưu embedding cho {name}")
    else:
        print("[ERROR] Không thu thập được embedding nào.")

# ==== NHẬN DIỆN KHUÔN MẶT REALTIME ====
def recognize():
    global last_results
    cap = cv2.VideoCapture(0)
    print("[INFO] Nhấn 'q' để thoát.")
    prev_time = time.time()
    frame_count = 0
    process_interval = 2

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("[DEBUG] Failed to read frame")
            break

        frame_count += 1
        img_rgb = preprocess_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
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
            face, pt1, pt2 = extract_face(img_rgb, res['box'])
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

            label = f"{name}" if name == "unknown" else f"{name} ({min_dist:.2f})"
            color = (0, 0, 255) if name == "unknown" else (0, 255, 0)
            cv2.rectangle(frame, pt1, pt2, color, 2)
            cv2.putText(frame, label, (pt1[0], pt1[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ==== MAIN ====
if __name__ == "__main__":
    print("=== HỆ THỐNG NHẬN DIỆN KHUÔN MẶT ===")
    if not os.path.exists(encodings_path):
        print("[INFO] encodings.pkl not found. Computing clustered embeddings...")
        encoding_dict = compute_clustered_embeddings(dataset_path, n_clusters=7)
    else:
        encoding_dict = load_pickle(encodings_path)
        print("[INFO] Loaded embeddings from encodings.pkl")

    if encoding_dict:
        print("[INFO] Tuning recognition threshold...")
        recognition_t = tune_threshold(encoding_dict, dataset_path, num_samples=100)
    else:
        print("[WARNING] No embeddings loaded. Using default threshold.")

    print("Nhập tên để đăng ký người mới (hoặc Enter để nhận diện luôn):")
    name = input("Tên người mới: ").strip()

    if name:
        register_new_face(name)
        print("[INFO] Recomputing clustered embeddings...")
        encoding_dict = compute_clustered_embeddings(dataset_path, n_clusters=7)

    recognize()
