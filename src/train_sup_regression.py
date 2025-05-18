import argparse
import torch
torch.cuda.empty_cache()
from tqdm import tqdm
from models.huggingface import Geolocalizer
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import pandas as pd
from dataset import CrossLocateUniformDatasetGeological
from math import radians, sin, cos, sqrt, atan2
import torch.nn.functional as F
import os
from heads import FullGeological

def haversine_loss(pred, target):
    """
    pred, target: tensors of shape (B, 2) in radians
    Returns mean haversine distance in kilometers.
    """
    R = 6371.0  # Earth radius in km

    dlat = pred[:, 0] - target[:, 0]
    dlon = pred[:, 1] - target[:, 1]

    a = torch.sin(dlat / 2) ** 2 + \
        torch.cos(pred[:, 0]) * torch.cos(target[:, 0]) * torch.sin(dlon / 2) ** 2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
    distance = R * c
    return distance.mean()


def forward_full(model, images, country, admin1, admin2):
    # ON CLASSES
    features = model.model.backbone.clip.vision_model(images).pooler_output

    reg_country = model.mid.reg_country(features).view(-1, 7, 2)
    reg_admin1 = model.mid.reg_admin1(features).view(-1, 58, 2)
    reg_admin2 = model.mid.reg_admin2(features).view(-1, 127, 2)

    batch_size = images.size(0)

    pred_c = reg_country[torch.arange(batch_size), country]
    pred_a1 = reg_admin1[torch.arange(batch_size), admin1]
    pred_a2 = reg_admin2[torch.arange(batch_size), admin2]

    return pred_c, pred_a1, pred_a2

def forward_admin2(model, images, admin2):
    # ON ADMIN2 CLASSES
    features = model.model.backbone.clip.vision_model(images).pooler_output
    reg_pred = model.mid.reg(features)

    reg_pred = reg_pred.view(-1, 127, 2)

    batch_size = admin2.size(0)
    selected_reg = reg_pred[torch.arange(batch_size), admin2]

    return selected_reg


def train_epoch(model, loader, optimizer, device, type_forward):
    model.train()
    total_loss = 0
    total_loss_har = 0

    for images, coords, cell, admin1, admin2, country in tqdm(loader, desc="Train"):
        images, coords, cell = images.to(device), coords.to(device), cell.to(device)
        admin1, admin2, country = admin1.to(device), admin2.to(device), country.to(device)

        if type_forward == "cca":
            reg_pred_c, reg_pred_a1, reg_pred_a2 = forward_full(model, images, country, admin1, admin2)
            loss = (
                reg_pred_c * 0.2 +
                reg_pred_a1 * 0.3 +
                reg_pred_a2 * 0.5
            )
            reg_loss = F.l1_loss(loss.float(), coords)
            reg_haversine = haversine_loss(loss, coords)
            loss = reg_loss


        if type_forward == "admin2":
            reg_pred = forward_admin2(model, images, admin2)
            reg_loss = F.mse_loss(reg_pred.float(), coords)
            reg_haversine = haversine_loss(reg_pred, coords)
            loss = reg_loss


        loss.backward()
        optimizer.step()
        total_loss += reg_loss.item()
        total_loss_har += reg_haversine.item()

    return total_loss / len(loader), total_loss_har / len(loader)

@torch.no_grad()
def eval_epoch(model, loader, device, type_forward):
    model.eval()
    total_loss = 0
    total_loss_har = 0

    for images, coords, cell, admin1, admin2, country in tqdm(loader, desc="Eval"):
        images, coords, cell = images.to(device), coords.to(device), cell.to(device)
        admin1, admin2, country = admin1.to(device), admin2.to(device), country.to(device)

        if type_forward == "cca":
            reg_pred_c, reg_pred_a1, reg_pred_a2 = forward_full(model, images, country, admin1, admin2)
            loss = (
                reg_pred_c * 0.2 +
                reg_pred_a1 * 0.3 +
                reg_pred_a2 * 0.5
            )
            reg_loss = F.l1_loss(loss.float(), coords)
            reg_haversine = haversine_loss(loss, coords)
            loss = reg_loss


        if type_forward == "admin2":
            reg_pred = forward_admin2(model, images, admin2)
            reg_loss = F.l1_loss(reg_pred.float(), coords)
            reg_haversine = haversine_loss(reg_pred, coords)
            loss = reg_loss


        total_loss += reg_loss.item()
        total_loss_har += reg_haversine.item()

    return total_loss / len(loader), total_loss_har / len(loader)


def train(type_forward, ckpt_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Geolocalizer.from_pretrained("osv5m/baseline")
    model = FullGeological(model)
    
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

    train_set = CrossLocateUniformDatasetGeological("../data/gt/train_classes_osm_full.csv", image_dir, transform=transform)
    print("Train set done.")
    
    val_set = CrossLocateUniformDatasetGeological("../data/gt/val_classes_osm_full.csv", image_dir, transform=transform)
    print("Validation set done.")
    
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=16,  shuffle=True, num_workers=2, pin_memory=True)

    print(f"Train and Validation sets loaded.")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    for epoch in range(10):
        train_loss, train_har = train_epoch(model, train_loader, optimizer, device, type_forward)
        val_loss, val_har = eval_epoch(model, val_loader, device, type_forward)
        print(f"Epoch {epoch}: Train Loss: {train_loss:.2f} | Val Loss: {val_loss:.2f}")
        print(f"Epoch {epoch} (Harvesine): Train Loss: {train_har:.2f} km | Val Loss: {val_har:.2f} km")

        # Save checkpoint
        torch.save(model.state_dict(), os.path.join(ckpt_path, f"osv5m_reg_epoch{epoch}.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate images from tar.gz using OSV5M baseline")
    parser.add_argument("--type_forward", default="caa", choices=['caa', 'admin2'], help="Path to input tar.gz archive")
    parser.add_argument("--ckpt_path", default="../checkpoints", help="Path where to store models")

    args = parser.parse_args()
    train(args.type_forward, args.ckpt_path)
