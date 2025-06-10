import json
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch.nn.functional as F
from rest_framework.decorators import api_view
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from PIL import Image
from attendance_app.models import SubjectDate, CustomUser, StudentAttendance
from attendance_app.permissions import IsTeacher, IsStudent
from rest_framework.permissions import IsAuthenticated
import os;
import cv2
import numpy as np
import mtcnn
import os
import pickle
import time
from hoa_newModel.architecture_embedding import InceptionResNetV2
from scipy.spatial.distance import cosine
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
from sklearn.preprocessing import Normalizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
confidence_t = 0.8
recognition_t = 0.4
required_size = (160, 160)

encodings_path = r'F:\codePython\PBL5\face_recognize\hoa_newModel\encodings1.pkl'
model_path = r'F:\codePython\PBL5\face_recognize\hoa_newModel\embedding_model_new6.h5'

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
    return img[y1:y2, x1:x2]

def get_embedding(face):
    try:
        face = cv2.resize(face, required_size)
    except:
        return None
    face = preprocess_input(face.astype(np.float32))
    encode = face_encoder.predict(np.expand_dims(face, axis=0), verbose=0)[0]
    return l2_normalizer.transform(encode.reshape(1, -1))[0]

class FaceRecognitionAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        uploaded_file = request.FILES.get("image")
        subject_date_id = request.data.get("subject_date_id")

        if not uploaded_file:
            return Response({"error": "No image provided"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            subject_date = SubjectDate.objects.get(id=subject_date_id)
        except SubjectDate.DoesNotExist:
            return Response({"boxes": [], "names": [], "scores": [], "attended_users": []}, status=status.HTTP_200_OK)

        if not subject_date.status:
            return Response({"boxes": [], "names": [], "scores": [], "attended_users": []}, status=status.HTTP_200_OK)

        try:
            image = Image.open(uploaded_file).convert('RGB')
            rgb_image = np.array(image)
        except Exception:
            return Response({"error": "Invalid image file"}, status=status.HTTP_400_BAD_REQUEST)

        results = {'boxes': [], 'names': [], 'scores': [], 'attended_users': []}
        attended_users = []

        encoding_dict = load_pickle(encodings_path)
        if not encoding_dict:
            return Response({"error": "No embedding data"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        faces = face_detector.detect_faces(rgb_image)

        for res in faces:
            if res['confidence'] < confidence_t:
                continue

            face = extract_face(rgb_image, res['box'])
            embedding = get_embedding(face)
            name = "Unknown"
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

            score = 1 - min_dist if name != "Unknown" else 0.0
            label = name

            if name != "Unknown":
                student = CustomUser.objects.filter(id=name).first()
                if student:
                    attendance = StudentAttendance.objects.filter(student=student, subject_date=subject_date).first()
                    if attendance:
                        if not attendance.status:
                            attendance.status = True
                            attendance.save()
                            label = student.username
                            attended_users.append(student.username)
                        else:
                            label = str(student.id) + " OK"
                    else:
                        label = "Unknown"
                else:
                    label = "Unknown"

            results['boxes'].append([x1, y1, x2, y2])
            results['names'].append(label)
            results['scores'].append(score)

        results['attended_users'] = attended_users
        return Response(results, status=status.HTTP_200_OK)

mtcnn_single = MTCNN(keep_all=False, device=device)
@api_view(['POST'])
def upload_face(request):
    student_id = request.data.get("student_id")
    image_file = request.FILES.get("image")

    if not student_id or not image_file:
        return Response({'error': 'Missing student_id or image'}, status=status.HTTP_400_BAD_REQUEST)

    # Tìm user
    try:
        user = CustomUser.objects.get(id=student_id)
    except CustomUser.DoesNotExist:
        return Response({'error': 'User not found'}, status=status.HTTP_404_NOT_FOUND)

    # Tạo thư mục lưu ảnh
    folder_name = str(user.id)
    base_dir = r'F:\codePython\PBL5\face_recognize\media\dataset'
    save_dir = os.path.join(base_dir, folder_name)
    os.makedirs(save_dir, exist_ok=True)

    try:
        img = Image.open(image_file).convert('RGB')
    except Exception as e:
        return Response({'error': f'Invalid image file: {str(e)}'}, status=status.HTTP_400_BAD_REQUEST)

    # Detect face box
    boxes, _ = mtcnn_single.detect(img)
    if boxes is None or len(boxes) == 0:
        return Response({'error': 'No face detected'}, status=status.HTTP_400_BAD_REQUEST)

    # Lấy box đầu tiên, crop và resize như collect.py
    x1, y1, x2, y2 = [int(v) for v in boxes[0]]
    face = img.crop((x1, y1, x2, y2)).resize((160, 160))

    # Tên ảnh = tên file gốc hoặc tạo tên mới nếu cần
    save_path = os.path.join(save_dir, image_file.name)
    face.save(save_path)

    return Response({'message': 'Face cropped and uploaded successfully'}, status=status.HTTP_200_OK)