import os
import pandas as pd

def transform_csv(df, with_cells, output_csv):
    with_cells["image"] = with_cells["image"].astype(str).str.strip()
    df["image"] = df["image"].astype(str).str.strip()

    image_map = {os.path.splitext(img)[0]: img for img in with_cells["image"]}
    
    df["image_base"] = df["image"]

    try:
        df = df[["image", "longitude", "latitude", "cluster_label_admin2", "image_base"]]
        df.rename(columns={"cluster_label_admin2": "admin_id"}, inplace=True)
    except KeyError as e:
        print(f"Column missing in the input data: {e}")
        return
    df = df[df["image_base"].isin(image_map)].copy()

    df["image"] = df["image_base"].map(image_map)

    df.drop("image_base", axis=1, inplace=True)

    df = df.merge(with_cells[["image", "cell_id"]], on="image", how="left")

    df.to_csv(output_csv, index=False)
    print(f"Transformed CSV saved to {output_csv}")

def classify():
    input_osm_csv = "../../data/osm_classes.csv"
    osm_df = pd.read_csv(input_osm_csv)


    output_csv = "../../data/gt/test_with_cells_osm.csv"
    test_csv = "../../data/gt/test_with_cells.csv"
    test_df = pd.read_csv(test_csv)
    transform_csv(osm_df, test_df, output_csv)

    output_csv = "../../data/gt/train_with_cells_osm.csv"
    train_csv = "../../data/gt/train_with_cells.csv"
    train_df = pd.read_csv(train_csv)
    transform_csv(osm_df, train_df, output_csv)

    output_csv = "../../data/gt/val_with_cells_osm.csv"
    val_csv = "../../data/gt/val_with_cells.csv"
    val_df = pd.read_csv(val_csv)
    transform_csv(osm_df, val_df, output_csv)


if __name__ == "__main__":
    classify()
