import argparse
import pandas as pd
import numpy as np

def preprocess(files):

    for i, file in enumerate(["train", "test", "val"]):

        data = np.load(files[i], allow_pickle=True).item()

        image_paths = data["qImageFns"]
        wgs_coords = data["qWGS"]  

        df = pd.DataFrame({
        "image": image_paths,
        "latitude": [coord[0] for coord in wgs_coords],
        "longitude": [coord[1] for coord in wgs_coords],
        })

        df.to_csv(f"../data/gt/{file}.csv", index=False)
        print(f"Saved {len(df)} entries to ../data/gt/{file}.csv")

    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate images from tar.gz using OSV5M baseline")
    parser.add_argument("--train_npy", default="../../data/gt/Alps_photos_to_depth_compact_train.npy", help="Path to output CSV file")
    parser.add_argument("--test_npy", default="../../data/gt/Alps_photos_to_depth_compact_test.npy", help="Path to output CSV file")
    parser.add_argument("--val_npy", default="../../data/gt/Alps_photos_to_depth_compact_val.npy", help="Path to output CSV file")
    
    args = parser.parse_args()
    preprocess([args.train_npy, args.test_npy, args.val_npy])
