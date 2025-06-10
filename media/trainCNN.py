# train.py
import os
import json
import torch
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from facenet_pytorch import InceptionResnetV1

# Config
device = 'cuda' if torch.cuda.is_available() else 'cpu'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'dataset')
OUT_FILE = r'F:\codePython\PBL5\face_recognize\recognize_app\centroids.json'
BATCH = 32

# DataLoader
overall_transform = transforms.Compose([
    transforms.Resize((160,160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3,[0.5]*3)
])
dataset = datasets.ImageFolder(DATA_DIR, transform=overall_transform)
loader = DataLoader(dataset, batch_size=BATCH, shuffle=False)

# Model
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Compute embeddings and centroids
emb_dict = {}
counts = {}
with torch.no_grad():
    for imgs, labels in loader:
        imgs = imgs.to(device)
        embs = model(imgs).cpu().numpy()
        for emb, lbl in zip(embs, labels.numpy()):
            name = dataset.classes[lbl]
            emb_dict.setdefault(name, []).append(emb)

centroids = {name: np.mean(v, axis=0).tolist() for name, v in emb_dict.items()}
with open(OUT_FILE, 'w') as f:
    json.dump(centroids, f)
print(f'[INFO] Saved {len(centroids)} centroids to {OUT_FILE}')