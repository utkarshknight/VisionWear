import os
import joblib
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

# ================= PATHS =================
MODEL_DIR = "models"
MODEL_PATH = "models/fashion_recommender.pkl"
METADATA_CSV = "fashion_metadata.csv"

os.makedirs(MODEL_DIR, exist_ok=True)

# ================= LOAD METADATA =================
metadata = pd.read_csv(METADATA_CSV)

# Fix Windows paths
if "image_path" in metadata.columns:
    metadata["image_path"] = metadata["image_path"].str.replace("\\", "/", regex=False)

# ================= FEATURE EXTRACTOR =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model = nn.Sequential(*list(model.children())[:-1])
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def extract_features(img_path):
    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(img)
    return features.squeeze().cpu().numpy()

# ================= FEATURE EXTRACTION =================
features = []
valid_rows = []

print("🔄 Extracting features...")

for _, row in metadata.iterrows():
    try:
        img_path = row["image_path"]
        if os.path.exists(img_path):
            features.append(extract_features(img_path))
            valid_rows.append(row)
    except:
        continue

features = np.array(features)
metadata = pd.DataFrame(valid_rows).reset_index(drop=True)

# ================= SAVE MODEL =================
joblib.dump(
    {
        "features": features,
        "metadata": metadata
    },
    MODEL_PATH
)

print("✅ Model trained and saved at:", MODEL_PATH)
print("📦 Total items:", len(metadata))
