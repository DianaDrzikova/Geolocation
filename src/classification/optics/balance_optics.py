import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

def reassign_noise_points(df):
    """
    Reassign points labeled -1 (noise) to nearest real cluster center.
    """

    # Separate valid clusters and noise
    valid_points = df[df['cluster_label'] >= 0]
    noise_points = df[df['cluster_label'] == -1]

    print(f"Found {len(noise_points)} noise points to reassign.")

    if len(noise_points) == 0:
        return df  # nothing to fix

    # Compute cluster centers
    centers = valid_points.groupby('cluster_label')[['latitude', 'longitude']].mean().reset_index()

    # Build KDTree
    tree = cKDTree(centers[['latitude', 'longitude']].values)

    # Query nearest center for each noise point
    _, nearest_center_idx = tree.query(noise_points[['latitude', 'longitude']].values)

    # Map nearest centers back to cluster labels
    reassigned_labels = centers.iloc[nearest_center_idx]['cluster_label'].values

    # Update noise points
    df.loc[df['cluster_label'] == -1, 'cluster_label'] = reassigned_labels

    print(f"Reassigned all noise points.")

    return df


def balance_clusters(df, min_size=100, max_size=300, eps_km=10):
    coords = df[['latitude', 'longitude']].values
    labels = df['cluster_label'].values

    unique_labels, counts = np.unique(labels, return_counts=True)
    large_clusters = unique_labels[counts > max_size]

    print(f"Splitting {len(large_clusters)} large clusters...")

    new_labels = labels.copy()
    next_cluster_id = labels.max() + 1

    for cluster_id in large_clusters:
        cluster_idx = np.where(labels == cluster_id)[0]
        cluster_points = coords[cluster_idx]

        if len(cluster_points) <= max_size:
            continue

        # Convert to radians for haversine
        radians_coords = np.radians(cluster_points)

        size_ratio = len(cluster_points) / max_size
        adjusted_eps_km = eps_km / np.sqrt(size_ratio)
        adjusted_eps_rad = adjusted_eps_km / 6371.0

        db = DBSCAN(eps=adjusted_eps_rad, min_samples=10, metric='haversine')
        sublabels = db.fit_predict(radians_coords)

        for i, idx in enumerate(cluster_idx):
            sublabel = sublabels[i]
            if sublabel == -1:
                new_labels[idx] = next_cluster_id  # assign noise to new cluster
                next_cluster_id += 1
            else:
                new_labels[idx] = next_cluster_id + sublabel

        next_cluster_id = new_labels.max() + 1

        df['cluster_label'] = new_labels


    # Count samples per cluster
    counts = df['cluster_label'].value_counts()
    small_clusters = counts[counts < min_size].index

    if len(small_clusters) == 0:
        print("No small clusters to merge.")
        return df

    print(f"Merging {len(small_clusters)} small clusters...")

    # Compute cluster centers
    centers = df.groupby('cluster_label')[['latitude', 'longitude']].mean()

    # Build tree on large clusters
    large_clusters = counts[counts >= min_size].index
    tree = cKDTree(centers.loc[large_clusters][['latitude', 'longitude']].values)

    # Reassign small-cluster points
    for cid in small_clusters:
        points = df[df['cluster_label'] == cid][['latitude', 'longitude']].values
        _, nearest_idx = tree.query(points)
        nearest_cids = large_clusters.values[nearest_idx]

        df.loc[df['cluster_label'] == cid, 'cluster_label'] = nearest_cids

    unique_ids = sorted(df["cluster_label"].unique())
    id_map = {old: new for new, old in enumerate(unique_ids)}
    df["cluster_label"] = df["cluster_label"].map(id_map)

    return df

def naive_balance(df):
    df = reassign_noise_points(df)
    df = balance_clusters(df)
    return df
