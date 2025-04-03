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

def evaluate(tar_path, output_csv):
    """Generate predictions from Osv5M model

    Args:
        tar_path (filename): path to input tar.gz archive
        output_csv (filename): path to output CSV file
    """

    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model from Hugging Face
    geolocalizer = Geolocalizer.from_pretrained("osv5m/baseline")
    geolocalizer.eval()
    geolocalizer.to(device)

    with tarfile.open(tar_path, "r:gz") as archive, open(output_csv, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['image', 'latitude_radians', 'longitude_radians'])  

        # Filter for image files
        valid_exts = ('.png', '.jpeg')
        members = [m for m in archive.getmembers() if m.name.lower().endswith(valid_exts)]

        for member in tqdm(members, desc="Evaluating images"):
            file = archive.extractfile(member)
            if file is None:
                continue

            image = Image.open(io.BytesIO(file.read())).convert("RGB")
            x = geolocalizer.transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                gps = geolocalizer(x).squeeze().tolist()  # [latitude, longitude] in radians

            writer.writerow([os.path.basename(member.name), gps[0], gps[1]])

    print(f"\nPredictions saved to '{output_csv}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate images from tar.gz using OSV5M baseline")
    parser.add_argument("--tar_path", required=True, help="Path to input tar.gz archive")
    parser.add_argument("--output_csv", required=True, help="Path to output CSV file")

    args = parser.parse_args()
    evaluate(args.tar_path, args.output_csv)
