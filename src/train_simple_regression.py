import argparse
import torch
from tqdm import tqdm
from models.huggingface import Geolocalizer
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import CrossLocateUniformDataset
import torch.nn.functional as F
import os
import argparse
from heads import SimpleRegression

def haversine_loss(pred, target):
    R = 6371.0 

    lat1, lon1 = pred[:, 0], pred[:, 1]
    lat2, lon2 = target[:, 0], target[:, 1]

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = torch.sin(dlat / 2)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2)**2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
    return (R * c).mean()

def forward(model, images):
    features = model.model.backbone.clip.vision_model(images).pooler_output
    reg_preds = model.mid.reg(features)

    return reg_preds


def train_epoch(model, loader, optimizer, device, loss_func):
    model.train()
    total_loss = 0

    for images, coords in tqdm(loader, desc="Train"):
        images, coords = images.to(device), coords.to(device)
        optimizer.zero_grad()

        reg_preds = forward(model, images)
        loss = loss_func(reg_preds, coords)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def eval_epoch(model, loader, device, loss_func):
    model.eval()
    total_loss = 0
    for images, coords in tqdm(loader, desc="Eval"):
        images, coords = images.to(device), coords.to(device)

        reg_preds = forward(model, images)
        loss = loss_func(reg_preds, coords)
        
        total_loss += loss.item()

    return total_loss / len(loader)

def train(loss, ckpt_path, freeze):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Geolocalizer.from_pretrained("osv5m/baseline")
    wrapper = SimpleRegression(model)
    model = wrapper.load().to(device)

    if freeze:
        for param in model.parameters():
            param.requires_grad = False

        for param in model.mid.reg.parameters():
            param.requires_grad = True

        print("Freezing..")

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

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    loss_func = F.l1_loss

    if loss == "haversine":
        loss_func = haversine_loss
  
    for epoch in range(10):
        train_loss = train_epoch(model, train_loader, optimizer, device, loss_func)
        val_loss = eval_epoch(model, val_loader, device, loss_func)

        if loss == "haversine":
            print(f"Epoch {epoch} (raw): Train Loss: {train_loss:.2f} km | Val Loss: {val_loss:.2f} km")
        else:
            print(f"Epoch {epoch} (raw): Train Loss: {train_loss:.2f} | Val Loss: {val_loss:.2f}")

        # Save checkpoint
        torch.save(model.state_dict(), os.path.join(ckpt_path, f"osv5m_reg_epoch{epoch}.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate images from tar.gz using OSV5M baseline")
    parser.add_argument("--loss", default="haversine", choices=['haversine', 'l1'], help="Path to input tar.gz archive")
    parser.add_argument("--ckpt_path", default="../checkpoints/srh_last", help="Path where to store models")
    parser.add_argument("--freeze", default=True, help="Path to input tar.gz archive")

    args = parser.parse_args()
    train(args.loss, args.ckpt_path, args.freeze)
