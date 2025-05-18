import argparse
import pandas as pd
import pycountry as pc
import os
import geopandas as gpd
import matplotlib.pyplot as plt


name_mapping1 = {
    # Switzerland
    "Valais": "Valais",
    "Grisons": "Graubünden",
    "Bern": "Bern",
    "Schwyz": "Schwyz",
    "Uri": "Uri",
    "Ticino": "Ticino",
    "Lucerne": "Lucerne",
    "Obwalden": "Obwalden",
    "Vaud": "Vaud",
    "Saint Gallen": "Sankt Gallen",
    "Glarus": "Glarus",
    "Zurich": "Zürich",
    "Thurgau": "Thurgau",
    "Basel-Landschaft": "Basel-Landschaft",
    "Fribourg": "Fribourg",
    "Appenzell Innerrhoden": "Appenzell Innerrhoden",
    "Appenzell Ausserrhoden": "Appenzell Ausserrhoden",
    "Zug": "Zug",
    "Neuchatel": "Neuchâtel",

    # Liechtenstein
    "Balzers": "Balzers",
    "Planken": "Planken",
    "Schellenberg": "Schellenberg",
    "Triesen": "Triesen",
    "Vaduz": "Valduz",  # GADM uses "Valduz"

    # Austria
    "Tyrol": "Tirol",
    "Lower Austria": "Niederösterreich",
    "Upper Austria": "Oberösterreich",
    "Salzburg": "Salzburg",
    "Carinthia": "Kärnten",
    "Styria": "Steiermark",
    "Vorarlberg": "Vorarlberg",

    # Germany
    "Bavaria": "Bayern",
    "Baden-Wuerttemberg": "Baden-Württemberg",

    # France
    "Rhone-Alpes": "Auvergne-Rhône-Alpes",  # modern name
    "Provence-Alpes-Cote d'Azur": "Provence-Alpes-Côte d'Azur",

    # Italy
    "Lombardy": "Lombardia",
    "Aosta Valley": "Valle d'Aosta",
    "Piedmont": "Piemonte",
    "Friuli Venezia Giulia": "Friuli-Venezia Giulia",
    "Trentino-Alto Adige": "Trentino-Alto Adige",
    "Veneto": "Veneto",

    # Slovenia (map municipalities to statistical regions)
    "Radece": "Savinjska",
    "Tolmin": "Goriška",
    "Skofja Loka": "Gorenjska",
    "Slovenska Bistrica": "Podravska",
    "Kranjska Gora": "Gorenjska",
    "Naklo": "Gorenjska",
    "Cerkno": "Goriška",
    "Maribor": "Podravska",
    "Gorenja Vas-Poljane": "Gorenjska",
    "Medvode": "Osrednjeslovenska",
    "Gorje": "Gorenjska",
    "Bovec": "Goriška",
    "Zelezniki": "Gorenjska",
    "Jezersko": "Gorenjska",
    "Solcava": "Savinjska",
    "Luce": "Savinjska",
}


def plot_distribution(df, path, level="admin1", size=(20, 10), log=False):

    plt.figure(figsize=size)

    if level == "admin1":
        x = df["NAME_1"]
    elif level == "admin2":
        x = df["NAME_2"]
    else:
        raise ValueError("Invalid level. Choose 'admin1' or 'admin2'.")

    y = df["area_km2"]

    # Sort by area
    plot_df = pd.DataFrame({"region": x, "area_km2": y}).sort_values("area_km2", ascending=False)

    bars = plt.bar(plot_df["region"], plot_df["area_km2"], color='skyblue', edgecolor='blue')

    plt.xlabel("Region")
    plt.ylabel("Area (km²)")
    plt.title(f"Area of {level.upper()} Regions")

    if log:
        plt.yscale("log")
        plt.ylabel("Area (km², log scale)")

    plt.xticks(rotation=90)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(path, f"{level}.png"))
    plt.close()

def alpha2_to_alpha3(alpha2_code):
    try:
        return pc.countries.get(alpha_2=alpha2_code).alpha_3
    except:
        return None


def area_real(osm_csv, geo_data):

    df = pd.read_csv(osm_csv)

    country_list = df['country'].unique()
    admin1_list = df['admin1'].unique()
    admin2_list = df['admin2'].unique()

    print(admin2_list)

    alpha3_codes = [alpha2_to_alpha3(code) for code in country_list]
    all_admin1 = []
    all_admin2 = []

    for iso3 in alpha3_codes:
        zip_path = os.path.join(geo_data, f"gadm41_{iso3}_shp.zip")

        if not os.path.exists(zip_path):
            print(f"Missing file: {zip_path}")
            continue

        # Read admin1
        try:
            gdf1 = gpd.read_file(f"zip://{zip_path}!gadm41_{iso3}_1.shp")
            gdf1 = gdf1.to_crs("EPSG:6933")  
            gdf1["area_km2"] = gdf1.geometry.area / 1e6
            gdf1["country"] = iso3
            all_admin1.append(gdf1[["NAME_1", "area_km2", "country"]])
            gdf1_list = gdf1['NAME_1'].unique()
        except Exception as e:
            print(f"Failed to read admin1 for {iso3}: {e}")

        # Read admin2
        try:
            if iso3 == "LIE":
                continue
            gdf2 = gpd.read_file(f"zip://{zip_path}!gadm41_{iso3}_2.shp")
            gdf2 = gdf2.to_crs("EPSG:6933")
            gdf2["area_km2"] = gdf2.geometry.area / 1e6
            gdf2["country"] = iso3
            all_admin2.append(gdf2[["NAME_2", "NAME_1", "area_km2", "country"]])
        except Exception as e:
            print(f"Failed to read admin2 for {iso3}: {e}")

    # Combine and save
    admin1_df = pd.concat(all_admin1)
    admin2_df = pd.concat(all_admin2)

    list2 = admin2_df["NAME_2"].unique()

    for i in range(len(list2)):
        print(list2[i])


    df_admin1 = df.copy()
    df_admin2 = df.copy()

    # ADMIN1
    original_names1 = list(name_mapping1.keys())
    mapped_names1 = [name_mapping1[name] for name in original_names1]

    df_admin1 = pd.DataFrame({
        "original_name": original_names1,
        "mapped_admin1": mapped_names1
    })

    df_admin1 = df_admin1.merge(admin1_df[["NAME_1", "area_km2", 'country']], left_on="mapped_admin1", right_on="NAME_1", how="left")

    df_admin1.drop(columns=["NAME_1"], inplace=True)
    df_admin1.to_csv(os.path.join(geo_data,"area_admin1.csv"), index=False)

    plot_distribution(admin1_df, geo_data, level="admin1")

def area_dataset(osm_csv):
    osm_df = pd.read_csv(osm_csv)

    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute area of admin regions")
    parser.add_argument("--osm_csv", default="../../data/osm_classes.csv", help="Path to output CSV file")
    parser.add_argument("--geo_data", default="../../data/admin_boundaries", help="Path to output CSV file")

    args = parser.parse_args()
    area_real(args.osm_csv, args.geo_data)
    area_dataset(args.osm_csv)

