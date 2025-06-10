import cv2
import numpy as np
import mtcnn
import os
import pickle
from architecture_embedding import InceptionResNetV2
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
import imgaug.augmenters as iaa

# Cấu hình
confidence_t = 0.8
required_size = (160, 160)
encodings_path = r'F:\codePython\PBL5\face_recognize\hoa_newModel\encodings1.pkl'
model_path = r'F:\codePython\PBL5\face_recognize\hoa_newModel\embedding_model_new6.h5'
dataset_path = r'F:\codePython\PBL5\face_recognize\media\dataset'
os.makedirs("encodings", exist_ok=True)

# Khởi tạo mô hình và detector
l2_normalizer = Normalizer('l2')
face_encoder = InceptionResNetV2()
face_encoder.load_weights(model_path)
face_detector = mtcnn.MTCNN()

# Augmentation pipeline
aug = iaa.Sequential([
    iaa.Sometimes(0.5, iaa.Affine(scale=(0.9, 1.1))),
    iaa.Sometimes(0.5, iaa.Fliplr(0.5)),
    iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.5))),
    iaa.Sometimes(0.5, iaa.Multiply((0.8, 1.2))),
    iaa.Sometimes(0.3, iaa.contrast.LinearContrast((0.8, 1.2))),
])

def load_pickle(path):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return {}

def save_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

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
                face = extract_face(img_rgb, res['box'])
                # Tạo augmented faces
                augmented_faces = aug.augment_images([face] * 3)
                for aug_face in [face] + augmented_faces:
                    embedding = get_embedding(aug_face)
                    if embedding is not None:
                        embeddings.append(embedding)
        if embeddings:
            kmeans = KMeans(n_clusters=min(n_clusters, len(embeddings)), random_state=0)
            kmeans.fit(embeddings)
            encoding_dict[name] = kmeans.cluster_centers_
            print(f"[INFO] Đã xử lý: {name} với {len(embeddings)} embeddings")
        else:
            print(f"[WARNING] Không tìm thấy embeddings cho: {name}")

    save_pickle(encoding_dict, encodings_path)
    print(f"[INFO] Đã lưu clustered embeddings vào {encodings_path}")
    return encoding_dict

if __name__ == "__main__":
    print("[INFO] Bắt đầu train embeddings từ dataset...")
    compute_clustered_embeddings(dataset_path)
    print("[INFO] Hoàn tất.")
