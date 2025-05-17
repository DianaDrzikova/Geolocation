import tarfile
import io
import csv
import os
import argparse
from PIL import Image
import torch
from tqdm import tqdm
import sys
from models.huggingface import Geolocalizer
import pandas as pd

def evaluate(tar_path, test_csv_path,output_csv, checkpoint_path, eval_type):
    """Generate predictions from Osv5M model

    Args:
        tar_path (filename): path to input tar.gz archive
        output_csv (filename): path to output CSV file
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model from Hugging Face
    geolocalizer = Geolocalizer.from_pretrained("osv5m/baseline")

    if checkpoint_path and eval_type != "baseline":
        if eval_type == "simple":
            geolocalizer.mid.reg = torch.nn.Sequential(
                torch.nn.LayerNorm(1024),
                torch.nn.Linear(1024, 512),
                torch.nn.GELU(),
                torch.nn.Linear(512, 2)
            )

        print(f"Loading checkpoint from: {checkpoint_path}")
        geolocalizer.load_state_dict(torch.load(checkpoint_path, map_location=device))

    geolocalizer.eval()
    geolocalizer.to(device)

    test_df = pd.read_csv(test_csv_path)
    image_names = set(test_df['image'].str.strip())

    headers = ['image', 'latitude_radians', 'longitude_radians']

    if eval_type == "original":
        headers.append("cell_id")

    with tarfile.open(tar_path, "r:gz") as archive, open(output_csv, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)

        #valid_exts = ('.png', '.jpeg', '.jpg', '.JPG', '.JPEG', '.PNG')

        members = [m for m in archive.getmembers()
                   if os.path.basename(m.name) in image_names]

        for member in tqdm(members, desc="Evaluating images"):
            file = archive.extractfile(member)
            if file is None:
                continue

            image = Image.open(io.BytesIO(file.read())).convert("RGB")
            x = geolocalizer.transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                
                if eval_type == "simple":
                    features = geolocalizer.model.backbone.clip.vision_model(x).pooler_output
                    gps = geolocalizer.mid.reg(features).squeeze().tolist()
                    row = [os.path.basename(member.name), gps[0], gps[1]]
                elif eval_type == "original":
                    features = geolocalizer.backbone({"img": x})
                    mid_out = geolocalizer.mid(features)
                    logits = geolocalizer.head.classif(mid_out)
                    cell_id = torch.argmax(logits, dim=-1).item()
                    out = geolocalizer.head(mid_out, cell_id)
                    gps = out["gps"]
                    row = [os.path.basename(member.name), gps[0], gps[1], cell_id]
                else:
                    gps = geolocalizer(x).squeeze().tolist()
                    row = [os.path.basename(member.name), gps[0], gps[1]]
            
            writer.writerow(row)

            
    print(f"\nPredictions saved to '{output_csv}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate images from tar.gz using OSV5M baseline")
    parser.add_argument("--tar_path", default="../../query_photos.tar.gz", help="Path to input tar.gz archive")
    parser.add_argument("--test_csv", default="../data/gt/test_with_cells.csv", help="Path to test CSV file")
    parser.add_argument("--output_csv", default="../data/results/original_head_full/finetuned_predictions9.csv", help="Path to output CSV file")
    parser.add_argument("--checkpoint_path", default="../checkpoints/original_head_full/osv5m_reg_epoch9.pth", help="Path to model checkpoint (.pth)")
    parser.add_argument("--eval_type", default="original", choices=['baseline', 'simple', 'original'], help="Type of evaluation.")

    args = parser.parse_args()
    print(args.checkpoint_path)
    evaluate(args.tar_path, args.test_csv, args.output_csv, args.checkpoint_path, args.eval_type)
