import numpy as np
import math
from geopy.distance import geodesic
import matplotlib.pyplot as plt
from math import radians, sin, cos, sqrt, atan2
import pandas as pd

def evaluate_geolocation_classification(df):
    df['latitude_radians'] = df['latitude_radians'].astype(float)
    df['longitude_radians'] = df['longitude_radians'].astype(float)
    
    df['pred_lat_deg'] = df['latitude_radians'] * (180.0 / math.pi)
    df['pred_lon_deg'] = df['longitude_radians'] * (180.0 / math.pi)

    distances = []
    for idx, row in df.iterrows():
        gt_coords = (row['latitude'], row['longitude'])  # in degrees
        pred_coords = (row['pred_lat_deg'], row['pred_lon_deg'])  # in degrees
        dist_m = geodesic(gt_coords, pred_coords).meters
        distances.append(dist_m)

    distances = np.array(distances)

    print("\nDistance stats:")
    if len(distances) > 0:
        print(f"  Min:    {distances.min():.2f} m")
        print(f"  Max:    {distances.max():.2f} m")
        print(f"  Mean:   {distances.mean():.2f} m")
        print(f"  Median: {np.median(distances):.2f} m")
    else:
        print("  No distances computed (0 rows).")

    return distances

def print_recall_at_thresholds(distances, label, thresholds=[1000, 25000, 200000, 300000, 340000]):
    print(f"\nRecall for {label}:")
    distances = np.array(distances)
    for t in thresholds:
        r = np.mean(distances <= t)
        print(f"  Recall @ {t/1000:.0f} km: {r:.3f}")


def plot_recall_curve(distances_list, labels, max_dist=400000, step=100, axvline=340_000):
    thresholds = np.arange(0, max_dist + step, step)
    for distances, label in zip(distances_list, labels):
        recall = [np.mean(np.array(distances) <= t) for t in thresholds]
        plt.plot(thresholds, recall, label=label)

    plt.axvline(x=axvline, color='black', linestyle=':', label='450km distance')
    plt.xlabel("Distance error (m)")
    plt.ylabel("Recall")
    plt.title("Localization (Uniform dataset)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000  # Earth radius in meters
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def compute_distances(gt_df, pred_df):
    distances = []
    for _, row in gt_df.iterrows():
        image = row['image']
        pred_row = pred_df[pred_df['image'].str.strip() == image.strip()]
        if not pred_row.empty:
            lat_gt, lon_gt = row['latitude'], row['longitude']
            lat_pred = np.degrees(pred_row.iloc[0]['latitude_radians'])
            lon_pred = np.degrees(pred_row.iloc[0]['longitude_radians'])
            d = haversine_distance(lat_gt, lon_gt, lat_pred, lon_pred)
            distances.append(d)
    return distances


def eval_error_distribution(dist_list, labels_list):
    for i, dist in enumerate(dist_list):
        plt.hist(dist, bins=50, alpha=0.25, label=labels_list[i])
    plt.xlabel('Distance error (m)')
    plt.ylabel('Number of images')
    plt.legend()
    plt.title('Error Distribution Before vs After Fine-tuning')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_loss(train_loss, val_loss, name):
    epochs = list(range(1, len(train_loss) + 1))

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label='Train Loss', marker='o')
    plt.plot(epochs, val_loss, label='Validation Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training vs Validation Loss ({name})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_acc(train_loss, val_loss, name):
    epochs = list(range(1, len(train_loss) + 1))

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label='Train Accuracy', marker='o')
    plt.plot(epochs, val_loss, label='Validation Accuracy', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy %')
    plt.title(f'Training vs Validation Accuracy ({name})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def evaluate_geolocation(df_gt, df_pred):

    df_pred.rename(columns={"image": "image_pred"}, inplace=True) # remove .jpg, .png, etc.

    # Strip file extension from the predictions
    df_pred['image_clean'] = (
        df_pred['image_pred']
        .str.replace(r'\.[^.]+$', '', regex=True)
        .str.strip()
    )

    df_gt['image'] = df_gt['image'].str.strip()
    df_pred['pred_lat_deg'] = df_pred['latitude_radians'] * (180.0 / math.pi)
    df_pred['pred_lon_deg'] = df_pred['longitude_radians'] * (180.0 / math.pi)

    df_merged = pd.merge(
        df_gt,
        df_pred,
        left_on='image',        # ground truth
        right_on='image_clean'  # predictions
    )

    distances = []
    min = np.inf 
    min_idx, max_idx = 0, 0
    max = -np.inf
    for idx, row in df_merged.iterrows():
        gt_coords = (row['latitude'], row['longitude'])             # degrees
        pred_coords = (row['pred_lat_deg'], row['pred_lon_deg'])    # degrees
        dist_m = geodesic(gt_coords, pred_coords).meters

        if dist_m < min:
            min = dist_m
            min_idx = idx 
        
        if dist_m > max: 
            max = dist_m
            max_idx = idx 

        distances.append(dist_m)

    print(df_merged.loc[min_idx])
    print(df_merged.loc[max_idx])

    distances = np.array(distances)

    print("\nDistance stats:")
    if len(distances) > 0:
        print(f"  Min: {distances.min():.2f} m")
        print(f"  Max: {distances.max():.2f} m")
        print(f"  Mean: {distances.mean():.2f} m")
        print(f"  Median: {np.median(distances):.2f} m")
    else:
        print("  No distances computed (0 rows).")
