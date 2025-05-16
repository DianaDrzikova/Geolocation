import pandas as pd
import numpy as np
from sklearn.cluster import OPTICS
from tqdm import tqdm

def evaluate_optics(df, min_samples, xi, target_clusters=80):
    coords = df[['latitude', 'longitude']].values

    optics = OPTICS(min_samples=min_samples, xi=xi)
    labels = optics.fit_predict(coords)

    labels = labels[labels != -1]

    if len(np.unique(labels)) == 0:
        return np.inf

    num_clusters = len(np.unique(labels))
    counts = np.bincount(labels)

    # Penalty if the number of clusters vary too much
    cluster_penalty = abs(num_clusters - target_clusters)

    # Penalty if the size of clusters is not proportional
    balance_penalty = np.std(counts) / np.mean(counts)

    # Final score (lower = better), balance penalty has more weight
    # as the main priority is to have evenly distributed samples in
    # the classes. 
    score = cluster_penalty + balance_penalty * 50

    return score, num_clusters, np.mean(counts), np.std(counts)

def search_best_optics(df, target_clusters=80):

    best_score = np.inf
    best_params = None

    results = []

    for min_samples in tqdm(range(10, 50, 5)):
        for xi in np.linspace(0.02, 0.1, 5): 
            score, num_clusters, mean_size, std_size = evaluate_optics(df, min_samples, xi, target_clusters)
            results.append((min_samples, xi, score, num_clusters, mean_size, std_size))

            if score < best_score:
                best_score = score
                best_params = (min_samples, xi)

    # Best OPTICS parameters: min_samples=40, xi=0.020
    print(f"\nBest OPTICS parameters: min_samples={best_params[0]}, xi={best_params[1]:.3f}")

    return best_params[0], best_params[1]

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess CrossLocate dataset with OPTICS clustering")
    parser.add_argument("--input_csv", required=True, help="Path to input CSV file")
    parser.add_argument("--target_clusters", type=int, default=80, help="Target number of clusters")

    args = parser.parse_args()
    search_best_optics(args.input_csv, args.target_clusters)