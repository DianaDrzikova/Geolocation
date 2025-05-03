import argparse
import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
from sklearn.cluster import KMeans

def haversine(lat1, lon1, lat2, lon2):
    """
    Compute the great-circle distance between two geographic coordinates.
    Used instead of Euclidean distance due to Earth's curvature.
    Returns distance in kilometers.
    """
    R = 6371  # km
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

def split_and_merge(coords, min_size=100, max_size=300, max_iter=10):
    """
    Perform KMeans-based clustering with min/max size constraints:
    - Recursively splits clusters that are too large
    - Merges clusters that are too small with nearby ones
    - Stops when no more merges/splits are needed
    """
    coords = np.array(coords)
    labels = np.full(len(coords), -1) # Initially all unassigned
    next_label = 0
    pending = [(coords, np.arange(len(coords)))] # Clusters to process

    # First pass: recursively split large groups
    while pending:
        cluster_coords, idxs = pending.pop()
        # If this group is within size limits, assign it directly
        if len(idxs) <= max_size:
            labels[idxs] = next_label
            next_label += 1
        else:
            # Too big: use KMeans to split into multiple smaller groups
            n_split = int(np.ceil(len(idxs) / max_size)) # Find appropriate number of clusters
            kmeans = KMeans(n_clusters=n_split, random_state=42).fit(cluster_coords)
            # Push each subcluster back into the stack to re-check recursively
            for sub_k in np.unique(kmeans.labels_):
                sub_idxs = idxs[kmeans.labels_ == sub_k]
                pending.append((coords[sub_idxs], sub_idxs))

    # Second pass: refine clusters to satisfy constraints
    for _ in range(max_iter):
        changed = False
        # Find current label set and their sizes
        unique_labels, counts = np.unique(labels, return_counts=True)
        # Compute the centroid of each label (mean of all its points)
        centroids = {l: coords[labels == l].mean(axis=0) for l in unique_labels}

        # Merge small clusters
        for label, count in zip(unique_labels, counts):
            if count < min_size:
                idxs = np.where(labels == label)[0]
                # Find nearby candidates eligible for merging (not too big)
                candidates = [(hl, haversine(*centroids[label], *centroids[hl]))
                              for hl in unique_labels if hl != label and min_size <= np.sum(labels == hl) <= (max_size - count)]
                if not candidates:
                    continue

                # Merge to the closest valid cluster
                nearest_cluster = None
                min_distance = float('inf')
                for candidate_cluster, distance in candidates:
                    if distance < min_distance:
                        min_distance = distance
                        nearest_cluster = candidate_cluster

                labels[idxs] = nearest_cluster
                changed = True

        # Split large clusters
        for label in np.unique(labels):
            idxs = np.where(labels == label)[0]
            if len(idxs) > max_size:
                n_split = int(np.ceil(len(idxs) / max_size)) # Find appropriate number of clusters
                kmeans = KMeans(n_clusters=n_split, random_state=42).fit(coords[idxs])
                # Assign new labels to each subcluster
                for sub_k in np.unique(kmeans.labels_):
                    labels[idxs[kmeans.labels_ == sub_k]] = next_label
                    next_label += 1
                changed = True

        if not changed:
            break

    # Normalize label indices
    final_unique = np.unique(labels)
    remap = {old: new for new, old in enumerate(final_unique)}
    return np.vectorize(remap.get)(labels)

def assign_clusters_to_df(df, group, admin1_name, cluster_idx):
    """
    Assign spatial clusters to one 'admin1' region and update the main DataFrame with global cluster IDs.
    """

    coords = group[['latitude', 'longitude']].values
    labels = split_and_merge(coords)
    label_map = {local: cluster_idx + i for i, local in enumerate(np.unique(labels))}
    global_labels = [label_map[l] for l in labels]
    global_names = [f"{admin1_name}#{label_map[l]}" for l in labels]

    # Assign to DataFrame
    df.loc[group.index, 'cluster_label'] = global_labels
    df.loc[group.index, 'cluster_label_name'] = global_names
    return df, cluster_idx + len(label_map)

def handle_small_admin1(df, admin1_name, group, cluster_idx):
    """
    Try to merge very small admin1 regions into nearby clusters.
    Fallback: assign a new unique cluster label.
    """
    lat1, lon1 = group[['latitude', 'longitude']].mean()
    df_so_far = df[df['cluster_label'].notna()]
    centroids = df_so_far.groupby('cluster_label')[['latitude', 'longitude']].mean()
    sizes = df_so_far['cluster_label'].value_counts()

    best_cluster, best_dist = None, float('inf')
    # Find the closest existing cluster centroid within 75 km that can accept new points
    for cid, (lat2, lon2) in centroids.iterrows():
        if sizes[cid] + len(group) > 300: # Skip if target cluster would exceed max size
            continue
        dist = haversine(lat1, lon1, lat2, lon2)
        if dist < best_dist and dist <= 75: # Keep track of the closest eligible cluster
            best_cluster = cid
            best_dist = dist

    if best_cluster is not None:
        # Merge the small group into the nearest cluster
        df.loc[group.index, 'cluster_label'] = best_cluster
        df.loc[group.index, 'cluster_label_name'] = f"{admin1_name}#merged_to_{best_cluster}"
    else:
        # No suitable cluster found - assign new cluster label to this region
        df.loc[group.index, 'cluster_label'] = cluster_idx
        df.loc[group.index, 'cluster_label_name'] = f"{admin1_name}#1"
        cluster_idx += 1
    
    return df, cluster_idx

def cluster_geolocations(input_csv, output_csv):
    """
    Cluster the full OSM geolocation DataFrame by region (admin1), ensuring:
    - cluster balance (100 ≤ size ≤ 300)
    - geographical coherence
    - small regions merged or labeled separately
    - distant outliers reassigned
    """

    df = pd.read_csv(input_csv)
    df['admin2'] = df['admin2'].fillna(df['admin1'])
    df['cluster_label'] = np.nan
    df['cluster_label_name'] = None
    cluster_idx = 0

     # Initial pass: split by admin1 group
    for admin1_name, group in df.groupby('admin1'):
        count = len(group)
        if count < 100: # Small cluster, needs merge
            df, cluster_idx = handle_small_admin1(df, admin1_name, group, cluster_idx)
        elif count <= 300: # Cluster size is within the range
            df.loc[group.index, 'cluster_label'] = cluster_idx
            df.loc[group.index, 'cluster_label_name'] = f"{admin1_name}#1"
            cluster_idx += 1
        else: # Big clusters, needs split
            df, cluster_idx = assign_clusters_to_df(df, group, admin1_name, cluster_idx)

    # Outlier reassignment based on large distance to cluster centroid
    centroids = df.groupby('cluster_label')[['latitude', 'longitude']].mean()
    df["distance_to_centroid"] = df.apply(
        lambda row: haversine(
            row["latitude"], row["longitude"],
            centroids.loc[row["cluster_label"]]["latitude"],
            centroids.loc[row["cluster_label"]]["longitude"]), axis=1)

    # Reassign samples too far from cluster centroid
    cluster_sizes = df['cluster_label'].value_counts().to_dict()
    outliers = df[df["distance_to_centroid"] > 100]

    for i, row in outliers.iterrows():
        lat, lon = row["latitude"], row["longitude"]
        best_cid, best_dist = None, float('inf')
        for cid, center in centroids.iterrows():
            if cluster_sizes[cid] >= 300: # If found cluster is too big, cannot assign
                continue
            dist = haversine(lat, lon, center["latitude"], center["longitude"])
            # Try to find new cluster to merge with based on maximum distance 75km^2
            if dist < best_dist and dist < 75:
                best_cid = cid
                best_dist = dist
        if best_cid is not None:
            df.at[i, "cluster_label"] = best_cid
            df.at[i, "cluster_label_name"] = f"reassigned_outlier_to_{int(best_cid)}"
            cluster_sizes[best_cid] += 1

    df.drop(columns=["distance_to_centroid"], inplace=True)
    df.to_csv(output_csv, index=False)
    
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess CrossLocate dataset with OPTICS clustering")
    parser.add_argument("--input_csv", default="../../data/osm_classes.csv", help="Path to input CSV file")
    parser.add_argument("--output_csv", default="../../data/balanced_classes.csv", help="Path to output CSV file")

    args = parser.parse_args()
    cluster_geolocations(args.input_csv, args.output_csv)
