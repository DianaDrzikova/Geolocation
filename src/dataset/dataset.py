import os
import pandas as pd
import torch
from math import radians
from PIL import Image
from torch.utils.data import Dataset

class CrossLocateUniformDatasetCells(Dataset):
    def __init__(self, csv_path, image_root, transform=None):
        self.df = pd.read_csv(csv_path)
        self.image_root = image_root
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_root, row["image"].strip())
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        # lat/lon in radians
        coords = torch.tensor([radians(row.latitude),
                               radians(row.longitude)],
                              dtype=torch.float32)
        # cell_id as integer
        cell = torch.tensor(row.cell_id, dtype=torch.long)

        return img, coords, cell
    

class CrossLocateUniformDatasetGeological(Dataset):
    def __init__(self, csv_path, image_root, transform=None):
        self.df = pd.read_csv(csv_path)
        self.image_root = image_root
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_root, row["image"].strip())
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        # lat/lon in radians
        coords = torch.tensor([radians(row.latitude),
                               radians(row.longitude)],
                              dtype=torch.float32)
        cell = torch.tensor(row.cell_id, dtype=torch.long)
        admin1 = torch.tensor(row.admin1_id, dtype=torch.long)
        admin2 = torch.tensor(row.admin2_id, dtype=torch.long)
        country = torch.tensor(row.country_id, dtype=torch.long)
        
        return img, coords, cell, admin1, admin2, country

    
class CrossLocateUniformDataset(Dataset):
    def __init__(self, csv_path, image_root, transform=None):
        self.df = pd.read_csv(csv_path)
        self.image_root = image_root
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_root, row["image"].strip())
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        coords = torch.tensor([radians(row.latitude),
                               radians(row.longitude)],
                              dtype=torch.float32)
        return img, coords