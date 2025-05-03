import pandas as pd
import numpy as np
import argparse
import os
from sklearn.cluster import OPTICS
import balance_optics as po

def optics(input_csv, output_path, min_samples=40, xi=0.02):
    """
    Run OPTICS clustering on latitude and longitude.
    """

    # Load the dataset
    df = pd.read_csv(input_csv)

    coords = df[['latitude', 'longitude']].values

    # Run OPTICS clustering
    optics = OPTICS(min_samples=min_samples, xi=xi)
    cluster_labels = optics.fit_predict(coords)
    
    df['cluster_label'] = cluster_labels

    df.to_csv(os.path.join(output_path,"optics_classes.csv"), index=False)

    df_naive = po.naive_balance(df)
    df_naive.to_csv(os.path.join(output_path,"optics_classes_naive.csv"), index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Preprocess CrossLocate dataset with OPTICS clustering")
    parser.add_argument("--input_csv", default="../../../data/datasetInfoClean.csv", help="Path to input csv")
    parser.add_argument("--output_path", default="../../../data", help="Path to output CSV file with clusters")
    parser.add_argument("--min_samples", type=int, default=42, help="Minimum samples per cluster")
    parser.add_argument("--xi", type=float, default=0.02, help="Xi parameter for OPTICS (cluster separation)")
    parser.add_argument("--max_region", type=int, default=300, help="Xi parameter for OPTICS (cluster separation)")


    args = parser.parse_args()
    optics(args.input_csv, args.output_path, args.min_samples, args.xi)
