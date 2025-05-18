import argparse, csv, json, math, os, pathlib, time

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import open_clip
from dataset import CrossLocateUniformDataset
import torch.nn.functional as F
from tqdm import tqdm
from heads import CLIPRegressorBasic, CLIPRegressorGeoClip
from geoclip import GeoCLIP

def haversine_torch(pred_rad: torch.Tensor, target_rad: torch.Tensor) -> torch.Tensor:
    R = 6371.0088  # mean Earth radius (km)
    dlat = pred_rad[:, 0] - target_rad[:, 0]
    dlon = pred_rad[:, 1] - target_rad[:, 1]
    a = torch.sin(dlat / 2) ** 2 + torch.cos(pred_rad[:, 0]) * torch.cos(target_rad[:, 0]) * torch.sin(dlon / 2) ** 2
    return 2 * R * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))

class HaversineLoss(nn.Module):
    def forward(self, pred_rad: torch.Tensor, target_rad: torch.Tensor) -> torch.Tensor:
        return haversine_torch(pred_rad, target_rad).mean()

def run_epoch(model, loader, criterion, optimizer, device, train=True):
    epoch_loss, haversine_km = 0.0, 0.0
    model.train(train)

    bar = tqdm(loader, desc="Train" if train else "Val", leave=False)

    for imgs, targets_rad in bar:     
        imgs = imgs.to(device).half()  
        targets_rad = targets_rad.to(device)             

        preds_rad = model(imgs)                    

        loss = criterion(preds_rad, targets_rad)

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            km = haversine_torch(preds_rad, targets_rad).sum().item()
            haversine_km += km
            epoch_loss   += loss.item() * imgs.size(0)

        bar.set_postfix(loss=f"{loss.item():.3f}", km=f"{km:.1f}")

    n = len(loader.dataset)
    return epoch_loss / n, haversine_km / n


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    criterion = HaversineLoss() if args.loss == "haversine" else F.l1_loss()
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)

    if args.model_type == "basic":
        # CLIP backbone (frozen)
        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="laion2b_s32b_b82k", device=device
        )
        clip_model.requires_grad_(False)
        clip_model.eval()
        model = CLIPRegressorBasic(clip_model).to(device)
        model.clip.half()  

    if args.model_type == "geoclip":
        # GeoCLIP backbone (frozen)
        gc = GeoCLIP()
        _, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14")
        gc.requires_grad_(False)
        gc.eval()    
        model = CLIPRegressorGeoClip(gc).to(device)
        model.gc.half()  

    scratch_dir = os.environ["SCRATCHDIR"]
    image_dir = os.path.join(scratch_dir, "query_photos")

    train_ds = CrossLocateUniformDataset(args.train_csv, image_dir, preprocess)
    val_ds   = CrossLocateUniformDataset(args.val_csv,   image_dir, preprocess)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size*2, shuffle=False, num_workers=2, pin_memory=True)
    
    history = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, _ = run_epoch(model, train_dl, criterion, optimizer, device, train=True)
        val_loss, val_km = run_epoch(model, val_dl,   criterion, optimizer, device, train=False)
        scheduler.step(val_loss)

        print(f"Epoch {epoch}/{args.epochs} | train loss {train_loss:.4f} | val loss {val_loss:.4f} | val km {val_km:.1f} | {(time.time()-t0):.1f}s")

        ckpt_path = os.path.join(args.out_dir, f"epoch_{epoch:02d}.pt")
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict()
        }, ckpt_path)

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_km": val_km})
        with open(os.path.join(args.out_dir, "history.json"), "w") as fp:
            json.dump(history, fp, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", default="../data/gt/train_classes_osm_full.csv")
    parser.add_argument("--val_csv", default="../data/gt/val_classes_osm_full.csv")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--loss", choices=["l1", "haversine"], default="haversine")
    parser.add_argument("--out_dir", type=str, default="checkpoints")
    parser.add_argument("--model_type", type=str, default="basic", choices=["basic", "geoclip"])
    args = parser.parse_args()

    train(args)