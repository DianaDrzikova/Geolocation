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
from collections import Counter
from heads import SimpleRegression, ClassificationCountry, ClassificationAdmin

def forward(model, images):
    features = model.model.backbone.clip.vision_model(images).pooler_output
    logits = model.mid.country(features)
    return logits

def compute_class_weights(df, num_classes, column):
    admin_ids = df[column]
    
    counts = Counter(admin_ids)
    total = sum(counts.values())

    weights = [0.0] * num_classes
    for i in range(num_classes):
        weights[i] = total / (counts[i] if counts[i] > 0 else 1)

    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    return weights_tensor

#### TRAINING ####

def train_country(loader, device, model, loss_fnc, optimizer):
    total_loss, total_correct, total_samples = 0, 0, 0

    for images, coords, cell, _, _, country in tqdm(loader, desc="Train"):
            images, coords, cell = images.to(device), coords.to(device), cell.to(device)
            country = country.to(device)

            logits = forward(model, images)
            loss = loss_fnc(logits, country)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_correct += (logits.argmax(dim=1) == country).sum().item()
            total_samples += country.size(0)

    return total_loss / len(loader), total_correct / total_samples

def train_admin1(loader, device, model, loss_fnc, optimizer):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0

    for images, coords, cell, admin1, _, _ in tqdm(loader, desc="Train"):
            images, coords, cell = images.to(device), coords.to(device), cell.to(device)
            admin1 = admin1.to(device)

            logits = forward(model, images)
            loss = loss_fnc(logits, admin1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_correct += (logits.argmax(dim=1) == admin1).sum().item()
            total_samples += admin1.size(0)

    return total_loss / len(loader), total_correct / total_samples


def train_epoch(model, loader, optimizer, device, loss_fnc, class_type):
    if class_type == "country":
        return train_country(loader, device, model, loss_fnc, optimizer)
    
    if class_type == "admin1":
        return train_admin1(loader, device, model, loss_fnc, optimizer)
    
#### EVALUATION ####

def eval_country(model, loader, device):
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0

    for images, coords, cell, _, _, country in tqdm(loader, desc="Eval"):
        images, coords, cell = images.to(device), coords.to(device), cell.to(device)
        country = country.to(device)
        
        logits = forward(model, images)
        loss = F.cross_entropy(logits, country)

        total_loss += loss.item()
        total_correct += (logits.argmax(dim=1) == country).sum().item()
        total_samples += country.size(0)

    return total_loss / len(loader), total_correct / total_samples

def eval_admin1(model, loader, device):
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0

    for images, coords, cell, admin1, _, _ in tqdm(loader, desc="Eval"):
        images, coords, cell = images.to(device), coords.to(device), cell.to(device)
        admin1 = admin1.to(device)
        
        logits = forward(model, images)
        loss = F.cross_entropy(logits, admin1)

        total_loss += loss.item()
        total_correct += (logits.argmax(dim=1) == admin1).sum().item()
        total_samples += admin1.size(0)

    return total_loss / len(loader), total_correct / total_samples


@torch.no_grad()
def eval_epoch(model, loader, device, class_type):

    if class_type =="country":
        return eval_country(model, loader, device)
    
    if class_type =="admin1":
        return eval_admin1(model, loader, device)


def train(class_type, ckpt_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Geolocalizer.from_pretrained("osv5m/baseline")

    model = SimpleRegression(model)
    model.load_state_dict(torch.load("../checkpoints/simple_regression/osv5m_reg_epoch9.pth", map_location=device))  

    out = 58
    if class_type == "country":
        out = 7
        model = ClassificationCountry(model).add_class(out)\
        
    if class_type == "admin1":
        model = ClassificationAdmin(model).add_class(out)\

    for param in model.parameters():
        param.requires_grad = False

    for param in model.mid.country.parameters():
        param.requires_grad = True

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
    
    train_df = pd.read_csv("../data/gt/train_classes_osm_full.csv")
    class_weights = compute_class_weights(train_df, out, class_type+"_id")
    class_weights = class_weights.to(device)
    loss_fn_train = torch.nn.CrossEntropyLoss(weight=class_weights)
    
    print("Train set done.")
    
    val_set = CrossLocateUniformDatasetGeological("../data/gt/val_classes_osm_full.csv", image_dir, transform=transform)

    print("Validation set done.")
    
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=16,  shuffle=True, num_workers=2, pin_memory=True)

    print(f"Train and Validation sets loaded.")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    for epoch in range(10):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, loss_fn_train)
        val_loss, val_acc = eval_epoch(model, val_loader, device)
        print(f"Epoch {epoch}: Train Loss: {train_loss:.2f} | Val Loss: {val_loss:.2f}")
        print(f"Epoch {epoch}: Train Accuracy={train_acc:.2%} | Val Accuracy={val_acc:.2%}")
        
        # Save checkpoint
        torch.save(model.state_dict(), os.path.join(ckpt_path, f"osv5m_class_epoch{epoch}.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate images from tar.gz using OSV5M baseline")
    parser.add_argument("--class_type", default="country", choices=["country", "admin1"], help="Path to input tar.gz archive")
    parser.add_argument("--ckpt_path", default="../checkpoints", help="Path where to store models")

    args = parser.parse_args()
    train(args.class_type, args.ckpt_path)
