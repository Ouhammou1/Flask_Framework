import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
from config import Config

# ---------------- DATASET ---------------- #
class PascalVOCDataset(Dataset):
    def __init__(self, root, split="train"):
        self.root = Path(root)
        self.image_dir = self.root / "JPEGImages"
        self.mask_dir = self.root / "SegmentationClass"

        split_file = self.root / "ImageSets/Segmentation" / f"{split}.txt"
        with open(split_file) as f:
            self.ids = [line.strip() for line in f]

        self.transform = transforms.Compose([
            transforms.Resize(Config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],
                                 std=[0.229,0.224,0.225])
        ])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img = Image.open(self.image_dir / f"{img_id}.jpg").convert("RGB")
        mask = Image.open(self.mask_dir / f"{img_id}.png")

        img = self.transform(img)
        mask = torch.from_numpy(
            np.array(mask.resize(Config.IMAGE_SIZE, Image.NEAREST))
        ).long()

        return img, mask

# ---------------- MODEL ---------------- #
class SimpleSegmentationModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, num_classes, 2, stride=2)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

# ---------------- PREDICT ---------------- #
def predict_image(model, image_path, device):
    image = Image.open(image_path).convert("RGB")
    original = np.array(image)

    transform = transforms.Compose([
        transforms.Resize(Config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ])

    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = torch.argmax(model(x), dim=1)[0].cpu().numpy()

    return original, pred
