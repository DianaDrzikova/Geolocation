import argparse
from PIL import Image
import torch
from tqdm import tqdm
from models.huggingface import Geolocalizer
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import pandas as pd
from dataset import CrossLocateUniformDataset
from math import radians, sin, cos, sqrt, atan2
import torch.nn.functional as F
import os


# 3. Fine-tune function
def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0

    for images, coords in tqdm(loader, desc="Train"):
        images, coords = images.to(device), coords.to(device)
        optimizer.zero_grad()
        features = model.model.backbone.clip.vision_model(images).pooler_output
        reg_preds = model.mid.reg(features)

        #loss = haversine_loss(reg_preds, coords)
        loss = F.l1_loss(reg_preds, coords)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total_loss = 0
    for images, coords in tqdm(loader, desc="Eval"):
        images, coords = images.to(device), coords.to(device)

        # Extract pooled features properly
        features = model.model.backbone.clip.vision_model(images).pooler_output

        # Predict (lat, lon)
        reg_preds = model.mid.reg(features)

        # Use L1 loss in radians
        loss = F.l1_loss(reg_preds, coords)
        total_loss += loss.item()

    return total_loss / len(loader)

# 4. Main
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Geolocalizer.from_pretrained("osv5m/baseline")

    model.mid.reg = torch.nn.Sequential(
        torch.nn.LayerNorm(1024),
        torch.nn.Linear(1024, 512),
        torch.nn.GELU(),
        torch.nn.Linear(512, 2)
    )
    
    model.to(device)

    print(f"Model loaded to: {device}.")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4815, 0.4578, 0.4082], std=[0.2686, 0.2613, 0.2758]),
    ])

    print("Transforms set.")

    scratch_dir = os.environ["SCRATCHDIR"]
    image_dir = os.path.join(scratch_dir, "query_photos")

    train_set = CrossLocateUniformDataset("../data/gt/train.csv", image_dir, transform=transform)
    print("Train set done.")
    
    val_set = CrossLocateUniformDataset("../data/gt/val.csv", image_dir, transform=transform)
    print("Validation set done.")
    
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=32,  shuffle=True, num_workers=2, pin_memory=True)

    print(f"Train and Validation sets loaded.")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    for epoch in range(10):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = eval_epoch(model, val_loader, device)
        print(f"Epoch {epoch}: Train Loss: {train_loss:.2f} km | Val Loss: {val_loss:.2f} km")

        # Save checkpoint
        torch.save(model.state_dict(), f"../checkpoints/osv5m_reg_epoch{epoch}.pth")


if __name__ == "__main__":
    train()
