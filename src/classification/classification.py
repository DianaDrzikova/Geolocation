import os
import argparse
import pandas as pd
from sklearn.cluster import KMeans
import reverse_geocoder as rg

def kmeans(input_csv, output_path, n=50):
    """
    Cluster dataset with kmeans method 
    """
    df = pd.read_csv(input_csv)
    X = df[['latitude', 'longitude']]
    kmeans = KMeans(n_clusters=n, random_state=42)
    df['cluster_label'] = kmeans.fit_predict(X)
    df.to_csv(os.path.join(output_path, "kmeans_classes.csv"), index=False)


def osm(input_csv, output_path):
    """
    Cluster dataset based on OpenStreetMap 
    """
    df = pd.read_csv(input_csv)
    coords = list(zip(df['latitude'], df['longitude']))
    results = rg.search(coords, mode=1)

    df['city'] = [r['name'] for r in results]
    df['admin1'] = [r['admin1'] for r in results] 
    df['admin2'] = [r['admin2'] for r in results] 
    df['country'] = [r['cc'] for r in results]

    administrative = sorted(df['admin1'].unique())
    administrative_to_class = {admin1: idx for idx, admin1 in enumerate(administrative)}
    df['cluster_label_admin1'] = df['admin1'].map(administrative_to_class)

    administrative = sorted(df['admin2'].unique())
    administrative_to_class = {admin2: idx for idx, admin2 in enumerate(administrative)}
    df['cluster_label_admin2'] = df['admin2'].map(administrative_to_class)

    country = sorted(df['country'].unique())
    country_to_class = {c: idx for idx, c in enumerate(country)}
    df['cluster_label_country'] = df['country'].map(country_to_class)

    city = sorted(df['city'].unique())
    country_to_city = {c: idx for idx, c in enumerate(city)}
    df['cluster_label_city'] = df['city'].map(country_to_city)

    df.to_csv(os.path.join(output_path, "osm_classes.csv"), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate images from tar.gz using OSV5M baseline")
    parser.add_argument("--input_csv", default="../../data/gt/datasetInfoClean.csv", help="Path to input csv")
    parser.add_argument("--output_path", default="../../data/", help="Path to input tar.gz archive")
    parser.add_argument("--method", type=str, required=True, help="Clustering method",
                                    choices=["kmeans",
                                            "osm"])
    parser.add_argument("--n", type=int, default=50, help="Number of clusters for k-means clustering")


    args = parser.parse_args()

    if args.method == "kmeans":
        kmeans(args.input_csv, args.output_path, args.n)
    
    if args.method == "osm":
        osm(args.input_csv, args.output_path)
        