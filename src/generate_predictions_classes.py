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
from heads import SimpleRegression, ClassificationCountry, ClassificationAdmin

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
            geolocalizer = SimpleRegression(geolocalizer).load()
        elif eval_type == "country":
            geolocalizer = ClassificationCountry(geolocalizer).load(7)
        elif eval_type == "admin1":
            geolocalizer = ClassificationAdmin(geolocalizer).load(58)

        print(f"Loading checkpoint from: {checkpoint_path}")
        geolocalizer.load_state_dict(torch.load(checkpoint_path, map_location=device))   

    geolocalizer.eval()
    geolocalizer.to(device)

    test_df = pd.read_csv(test_csv_path)

    if eval_type == "admin1":
        admin_map = dict(zip(test_df['image'], test_df['admin1_id']))
        image_names = set(admin_map.keys())
    elif eval_type == "country":
        country_map = dict(zip(test_df['image'], test_df['country_id']))
        image_names = set(country_map.keys())

    headers = ['image', 'longitude_radians', 'latitude_radians', 'predicted_class', 'ground_truth_class']

    with tarfile.open(tar_path, "r:gz") as archive, open(output_csv, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)

        members = [m for m in archive.getmembers()
                   if os.path.basename(m.name) in image_names]

        for member in tqdm(members, desc="Evaluating images"):
            file = archive.extractfile(member)
            if file is None:
                continue

            filename = os.path.basename(member.name)
            image = Image.open(io.BytesIO(file.read())).convert("RGB")
            x = geolocalizer.transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                features = geolocalizer.model.backbone.clip.vision_model(x).pooler_output
                gps = geolocalizer.mid.reg(features).squeeze().tolist()
                
                if eval_type == "country":
                    country_id = country_map[filename]
                    pred_class = geolocalizer.mid.country(features)
                    pred_class_id = pred_class.argmax(dim=1).item() 
                    row = [filename, gps[0], gps[1], pred_class_id, country_id]
                elif eval_type == "admin1":
                    admin_id = admin_map[filename]
                    pred_class = geolocalizer.mid.admin(features)
                    pred_class_id = pred_class.argmax(dim=1).item() 
                    row = [filename, gps[0], gps[1], pred_class_id, admin_id]
                else:
                    gps = geolocalizer(x).squeeze().tolist()
                    row = [filename, gps[0], gps[1]]

            writer.writerow(row)

            
    print(f"\nPredictions saved to '{output_csv}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate images from tar.gz using OSV5M baseline")
    parser.add_argument("--tar_path", default="../../query_photos.tar.gz", help="Path to input tar.gz archive")
    parser.add_argument("--test_csv", default="../data/gt/test_with_cells_osm_full.csv", help="Path to test CSV file")
    parser.add_argument("--output_csv", default="../data/results/admin1_class/finetuned_predictions4.csv", help="Path to output CSV file")
    parser.add_argument("--checkpoint_path", default="../checkpoints/admin1_class/osv5m_reg_epoch4.pth", help="Path to model checkpoint (.pth)")
    parser.add_argument("--eval_type", default="country", choices=['simple', 'country', 'admin1'], help="Type of evaluation.")

    args = parser.parse_args()
    print(args.checkpoint_path)
    evaluate(args.tar_path, args.test_csv, args.output_csv, args.checkpoint_path, args.eval_type)
