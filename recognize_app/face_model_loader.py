# recognize_app/face_model_loader.py

import torch
import torch.nn.functional as F
import json
from facenet_pytorch import MTCNN, InceptionResnetV1

device = 'cuda' if torch.cuda.is_available() else 'cpu'
THRESHOLD = 0.8

# Load MTCNN & InceptionResnetV1
mtcnn = MTCNN(keep_all=True, device=device)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Load centroids
with open('F:\codePython\PBL5\face_recognize\recognize_app', 'r') as f:
    cent_dict = json.load(f)

names = list(cent_dict.keys())
centroids = torch.stack([F.normalize(torch.tensor(cent_dict[n]), dim=0) for n in names]).to(device)
