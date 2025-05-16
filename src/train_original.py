import argparse
import torch
torch.cuda.empty_cache()
from tqdm import tqdm
from models.huggingface import Geolocalizer
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import CrossLocateUniformDatasetCells
import torch.nn.functional as F
import os

def forward(model, img, cell_id):
    features = model.backbone({"img": img})  
    mid_out = model.mid(features) 
    out = model.head(mid_out, cell_id)
    gps = out["gps"]  
    return gps

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0

    for images, coords, cell in tqdm(loader, desc="Train"):
        images, coords, cell = images.to(device), coords.to(device), cell.to(device)
        optimizer.zero_grad()

        gps = forward(model, images, cell)
        loss = F.l1_loss(gps, coords)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total_loss = 0
    for images, coords, cell in tqdm(loader, desc="Eval"):
        images, coords, cell = images.to(device), coords.to(device), cell.to(device)

        gps = forward(model, images, cell)

        loss = F.l1_loss(gps, coords)
        total_loss += loss.item()

    return total_loss / len(loader)

def train(freeze):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Geolocalizer.from_pretrained("osv5m/baseline")

    if freeze:
        for param in model.parameters():
            param.requires_grad = False

        for param in model.mid.reg.parameters():
            param.requires_grad = True

    model.to(device)
    print("model.model.mid:", model.model.mid)
    print("model.model.head:", model.model.head)
    print(model)


    print(f"Model loaded to: {device}.")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4815, 0.4578, 0.4082], std=[0.2686, 0.2613, 0.2758]),
    ])

    print("Transforms set.")

    scratch_dir = os.environ["SCRATCHDIR"]
    image_dir = os.path.join(scratch_dir, "query_photos")

    train_set = CrossLocateUniformDatasetCells("../data/gt/train_classes.csv", image_dir, transform=transform)
    print("Train set done.")
    
    val_set = CrossLocateUniformDatasetCells("../data/gt/val_classes.csv", image_dir, transform=transform)
    print("Validation set done.")
    
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=16,  shuffle=True, num_workers=2, pin_memory=True)

    print(f"Train and Validation sets loaded.")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    for epoch in range(10):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = eval_epoch(model, val_loader, device)
        print(f"Epoch {epoch}: Train Loss: {train_loss:.2f} km | Val Loss: {val_loss:.2f} km")

        torch.save(model.state_dict(), f"../checkpoints/osv5m_reg_epoch{epoch}.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate images from tar.gz using OSV5M baseline")
    parser.add_argument("--freeze", default=False, help="Path to input tar.gz archive")

    args = parser.parse_args()
    train(args.freeze)
