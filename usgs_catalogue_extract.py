import requests
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from io import StringIO

# Configuration
# JJI Transmitter (Ebino, Japan)
JJI_LAT = 32.08
JJI_LON = 130.82

# 8 UltraMSK Receiver Stations
STATIONS = {
    "AKT": (39.75, 140.10),
    "ANA": (36.24, 137.98),
    "IMZ": (34.92, 136.98),
    "KMK": (43.81, 144.17),
    "KTU": (39.13, 141.49),
    "NSB": (43.57, 145.60),
    "STU": (33.57, 131.37),
    "TYH": (34.73, 138.98),
}

# Square parameters
N_POINTS    = 100       
SQUARE_KM   = 100       

# USGS parameters
START_DATE  = "2014-01-01"
END_DATE    = "2017-02-28"
MIN_MAG     = 3.0

# Broad bounding box for USGS query
MAX_LAT     = 48.0
MIN_LON     = 125.0
MAX_LON     = 150.0
MIN_LAT = 10.0

OUTPUT_DIR  = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


#Helper Functions

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = (np.sin(dphi/2)**2 +
         np.cos(phi1) * np.cos(phi2) * np.sin(dlam/2)**2)
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def intermediate_point(lat1, lon1, lat2, lon2, fraction):

    lat1, lon1 = np.radians(lat1), np.radians(lon1)
    lat2, lon2 = np.radians(lat2), np.radians(lon2)

    d = 2 * np.arcsin(np.sqrt(
        np.sin((lat2 - lat1) / 2)**2 +
        np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2)**2
    ))

    if d == 0:
        return np.degrees(lat1), np.degrees(lon1)

    A = np.sin((1 - fraction) * d) / np.sin(d)
    B = np.sin(fraction * d) / np.sin(d)

    x = A * np.cos(lat1) * np.cos(lon1) + B * np.cos(lat2) * np.cos(lon2)
    y = A * np.cos(lat1) * np.sin(lon1) + B * np.cos(lat2) * np.sin(lon2)
    z = A * np.sin(lat1) + B * np.sin(lat2)

    lat_p = np.degrees(np.arctan2(z, np.sqrt(x**2 + y**2)))
    lon_p = np.degrees(np.arctan2(y, x))

    return lat_p, lon_p


def km_to_degrees(km, lat):
    lat_deg = km / 111.0
    lon_deg = km / (111.0 * np.cos(np.radians(lat)))
    return lat_deg, lon_deg


def get_square_bounds(centre_lat, centre_lon, km):
    half_km = km / 2.0
    lat_offset, lon_offset = km_to_degrees(half_km, centre_lat)

    min_lat = centre_lat - lat_offset
    max_lat = centre_lat + lat_offset
    min_lon = centre_lon - lon_offset
    max_lon = centre_lon + lon_offset

    return min_lat, max_lat, min_lon, max_lon


def point_in_square(eq_lat, eq_lon, min_lat, max_lat, min_lon, max_lon):
    return (min_lat <= eq_lat <= max_lat and
            min_lon <= eq_lon <= max_lon)


#Generate 100 Points Along Each Path

def generate_path_points(tx_lat, tx_lon, rx_lat, rx_lon, n_points):
    points = []
    for i in range(n_points):
        # Centre of segment i
        fraction = (2 * i + 1) / (2 * n_points)
        lat_p, lon_p = intermediate_point(
            tx_lat, tx_lon, rx_lat, rx_lon, fraction
        )
        points.append((lat_p, lon_p))
    return points


def generate_all_path_points():
    print("Generating path points...")
    print(f"  {N_POINTS} points per path × 8 stations = "
          f"{N_POINTS * 8} total points")
    print(f"  Square size: {SQUARE_KM} km × {SQUARE_KM} km\n")

    all_points = {}
    print(f"  {'Station':<8} {'Distance (km)':>15} "
          f"{'Segment length (km)':>20}")
    print("  " + "-"*45)

    for stn, (rx_lat, rx_lon) in STATIONS.items():
        dist_m = haversine(JJI_LAT, JJI_LON, rx_lat, rx_lon)
        dist_km = dist_m / 1000
        seg_km = dist_km / N_POINTS

        points = generate_path_points(
            JJI_LAT, JJI_LON, rx_lat, rx_lon, N_POINTS
        )
        all_points[stn] = points

        print(f"  {stn:<8} {dist_km:>15.1f} {seg_km:>20.1f}")

    return all_points


# Download USGS Catalogue 

def download_usgs_catalogue():
    print("\nDownloading USGS earthquake catalogue...")
    print(f"  Region : {MIN_LAT}°N–{MAX_LAT}°N, "
          f"{MIN_LON}°E–{MAX_LON}°E")
    print(f"  Period : {START_DATE} to {END_DATE}")
    print(f"  Min Mag: M ≥ {MIN_MAG}")

    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    params = {
        "format":       "csv",
        "starttime":    START_DATE,
        "endtime":      END_DATE,
        "minmagnitude": MIN_MAG,
        "minlatitude":  MIN_LAT,
        "maxlatitude":  MAX_LAT,
        "minlongitude": MIN_LON,
        "maxlongitude": MAX_LON,
        "orderby":      "time",
    }

    try:
        response = requests.get(url, params=params, timeout=120)
        response.raise_for_status()
        df = pd.read_csv(StringIO(response.text))
        print(f"  Downloaded {len(df):,} earthquakes from USGS")
        return df
    except requests.exceptions.RequestException as e:
        print(f"  ERROR: {e}")
        return None


def clean_catalogue(df):
    cols = ["time", "latitude", "longitude", "depth", "mag",
            "magType", "place", "type"]
    cols_available = [c for c in cols if c in df.columns]
    df = df[cols_available].copy()

    # Keep only tectonic earthquakes
    if "type" in df.columns:
        df = df[df["type"] == "earthquake"].copy()

    df["time"] = pd.to_datetime(df["time"], utc=True)
    df["date"] = df["time"].dt.date
    df = df.dropna(subset=["latitude", "longitude", "mag"])
    df = df.rename(columns={
        "latitude":  "lat",
        "longitude": "lon",
        "mag":       "magnitude",
        "depth":     "depth_km"
    })

    # Add unique ID for deduplication later
    df = df.reset_index(drop=True)
    df["eq_id"] = df.index

    print(f"  After cleaning: {len(df):,} earthquakes")
    return df


#Filter by Squares

def filter_by_squares(df, all_points):
    print("\nFiltering earthquakes using square grid method...")
    print(f"  Checking {len(df):,} earthquakes against "
          f"{N_POINTS * len(STATIONS):,} squares...")

    # Collect all matched earthquake IDs
    matched_ids = set()

    # Track
    eq_paths = {}

    total_squares = 0

    for stn, points in all_points.items():
        stn_matched = 0

        for point_lat, point_lon in points:
            # Get square boundaries for this point
            min_lat, max_lat, min_lon, max_lon = get_square_bounds(
                point_lat, point_lon, SQUARE_KM
            )
            total_squares += 1
            mask = (
                (df["lat"] >= min_lat) & (df["lat"] <= max_lat) &
                (df["lon"] >= min_lon) & (df["lon"] <= max_lon)
            )
            matched = df[mask]["eq_id"].values

            for eq_id in matched:
                if eq_id not in eq_paths:
                    eq_paths[eq_id] = []
                if stn not in eq_paths[eq_id]:
                    eq_paths[eq_id].append(stn)

            matched_ids.update(matched)
            stn_matched += len(matched)

        print(f"  {stn:<8} → {stn_matched:,} matches "
              f"(before deduplication)")

    # Build filtered dataframe
    df_filtered = df[df["eq_id"].isin(matched_ids)].copy()
    df_filtered["matched_paths"] = df_filtered["eq_id"].map(
        lambda x: ",".join(eq_paths.get(x, []))
    )

    print(f"\n  Total squares checked  : {total_squares:,}")
    print(f"  Before deduplication   : {sum(len(v) for v in eq_paths.values()):,}")
    print(f"  After deduplication    : {len(df_filtered):,} unique earthquakes")
    print(f"  Removed as duplicates  : "
          f"{sum(len(v) for v in eq_paths.values()) - len(df_filtered):,}")

    return df_filtered


# Summary

def print_summary(df):
    """Print summary statistics."""
    print("\n" + "="*55)
    print("FILTERED CATALOGUE SUMMARY")
    print("="*55)
    print(f"  Total earthquakes      : {len(df):,}")
    print(f"  Unique earthquake days : {df['date'].nunique():,}")
    print(f"  Date range             : "
          f"{df['date'].min()} to {df['date'].max()}")
    print(f"  Magnitude range        : "
          f"{df['magnitude'].min():.1f} – {df['magnitude'].max():.1f}")
    print(f"  Mean magnitude         : {df['magnitude'].mean():.2f}")

    print("\n  Magnitude distribution:")
    bins   = [3.0, 4.0, 5.0, 6.0, 7.0, 10.0]
    labels = ["M3.0–3.9", "M4.0–4.9", "M5.0–5.9",
              "M6.0–6.9", "M7.0+"]
    df["mag_bin"] = pd.cut(df["magnitude"], bins=bins,
                           labels=labels, right=False)
    for label, count in (df["mag_bin"].value_counts()
                         .sort_index().items()):
        bar = "█" * (count // 20)
        print(f"    {label} : {count:>5,}  {bar}")

    print("\n  Earthquakes per year:")
    df["year"] = pd.to_datetime(
        df["date"].astype(str)).dt.year
    for year, count in df.groupby("year").size().items():
        print(f"    {year} : {count:,}")


# Visualise

def visualise(df_all, df_filtered, all_points):
    print("\nGenerating visualisation...")

    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    colors = {
        "AKT": "#2E6DA4", "ANA": "#2E7D52", "IMZ": "#C8922A",
        "KMK": "#1C7293", "KTU": "#B45309", "NSB": "#7B2D8B",
        "STU": "#C0392B", "TYH": "#5B8DB8"
    }

    XLIM = (125, 150)
    YLIM = (25, 48)

    proj = ccrs.PlateCarree()
    fig, axes = plt.subplots(
        1, 2, figsize=(18, 8),
        subplot_kw={"projection": proj}
    )
    fig.patch.set_facecolor("white")

    for ax_idx, ax in enumerate(axes):

        #Cartopy background
        ax.set_extent([XLIM[0], XLIM[1], YLIM[0], YLIM[1]],
                      crs=proj)
        ax.add_feature(cfeature.OCEAN, color="#C8DCF0", zorder=0)
        ax.add_feature(cfeature.LAND,  color="#D6E8D0", zorder=1)
        ax.add_feature(cfeature.COASTLINE,
                       edgecolor="#AACBA8",
                       linewidth=0.6, zorder=2)
        ax.add_feature(cfeature.BORDERS,
                       edgecolor="#BBBBBB",
                       linewidth=0.4, zorder=2)
        ax.gridlines(draw_labels=True, linewidth=0.4,
                     color="gray", alpha=0.4,
                     xlocs=range(125, 151, 5),
                     ylocs=range(25, 49, 5))

        if ax_idx == 0:
            stn_example = "AKT"
            for point_lat, point_lon in all_points[stn_example]:
                min_lat, max_lat, min_lon, max_lon = get_square_bounds(
                    point_lat, point_lon, SQUARE_KM
                )
                rect = patches.Rectangle(
                    (min_lon, min_lat),
                    max_lon - min_lon,
                    max_lat - min_lat,
                    linewidth=0.4,
                    edgecolor=colors[stn_example],
                    facecolor=colors[stn_example],
                    alpha=0.12, zorder=3,
                    transform=proj
                )
                ax.add_patch(rect)

            # Path lines all stations
            for stn, (rx_lat, rx_lon) in STATIONS.items():
                path_lats = [pt[0] for pt in all_points[stn]]
                path_lons = [pt[1] for pt in all_points[stn]]
                ax.plot(path_lons, path_lats,
                        color=colors[stn], linewidth=1.2,
                        alpha=0.85, zorder=4,
                        label=f"{stn} path",
                        transform=proj)

            # All earthquakes
            sc = ax.scatter(df_all["lon"], df_all["lat"],
                            c=df_all["magnitude"],
                            cmap="YlOrRd", s=6, alpha=0.5,
                            vmin=3.0, vmax=7.5, zorder=5,
                            transform=proj)
            title = (f"All M≥{MIN_MAG} Japan Earthquakes\n"
                     f"({len(df_all):,} events) — "
                     f"squares shown for {stn_example} path")

        else:
            # Path lines
            for stn, (rx_lat, rx_lon) in STATIONS.items():
                path_lats = [pt[0] for pt in all_points[stn]]
                path_lons = [pt[1] for pt in all_points[stn]]
                ax.plot(path_lons, path_lats,
                        color=colors[stn], linewidth=1.2,
                        alpha=0.85, zorder=4,
                        transform=proj)

            # Filtered earthquakes
            sc = ax.scatter(df_filtered["lon"], df_filtered["lat"],
                            c=df_filtered["magnitude"],
                            cmap="YlOrRd", s=10, alpha=0.7,
                            vmin=3.0, vmax=7.5, zorder=5,
                            transform=proj)
            title = (f"After Square Filtering\n"
                     f"({len(df_filtered):,} unique earthquakes)")

        # JJI transmitter
        ax.scatter(JJI_LON, JJI_LAT, marker="^", s=220,
                   color="#1B3A5C", zorder=6, transform=proj)
        ax.annotate("JJI", xy=(JJI_LON, JJI_LAT),
                    xycoords=proj._as_mpl_transform(ax),
                    xytext=(6, 5), textcoords="offset points",
                    fontsize=8.5, color="#1B3A5C",
                    fontweight="bold", zorder=7)

        # Receiver stations
        for stn, (lat, lon) in STATIONS.items():
            ax.scatter(lon, lat, marker="v", s=90,
                       color=colors[stn], zorder=6,
                       edgecolors="white", linewidths=0.5,
                       transform=proj)
            ax.annotate(stn, xy=(lon, lat),
                        xycoords=proj._as_mpl_transform(ax),
                        xytext=(5, 3), textcoords="offset points",
                        fontsize=7.5, color=colors[stn],
                        fontweight="bold", zorder=7)

        ax.set_title(title, fontsize=11, fontweight="bold",
                     color="#1B3A5C", pad=8)

        if ax_idx == 1:
            plt.colorbar(sc, ax=ax,
                         label="Magnitude", shrink=0.75)

        if ax_idx == 0:
            ax.legend(fontsize=7, loc="upper left",
                      framealpha=0.85, ncol=2,
                      edgecolor="#CCCCCC")

    plt.suptitle(
        f"USGS Catalogue — Square Grid Method "
        f"({N_POINTS} points × {SQUARE_KM}km squares per path)\n"
        f"JJI Transmitter (22.2 kHz) | "
        f"M≥{MIN_MAG} | {START_DATE} to {END_DATE}",
        fontsize=11, fontweight="bold", color="#1B3A5C", y=1.01
    )
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "square_map.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close()
    print(f"  Map saved: {out_path}")


# Main

def main():
    print("=" * 55)
    print("USGS CATALOGUE (SQUARE GRID METHOD)")
    print("=" * 55)

    # 1. Generate 100 points along each path
    all_points = generate_all_path_points()

    # 2. Download USGS catalogue
    df_raw = download_usgs_catalogue()
    if df_raw is None:
        print("Download failed. Check internet connection.")
        return

    # 3. Clean catalogue
    df_clean = clean_catalogue(df_raw)

    # 4. Filter by squares
    df_filtered = filter_by_squares(df_clean, all_points)

    # 5. Summary
    print_summary(df_filtered)

    # 6. Visualise
    visualise(df_clean, df_filtered, all_points)

    # 7. Save outputs
    # Full filtered catalogue
    cat_path = os.path.join(OUTPUT_DIR,
                            "usgs_catalogue_squares.csv")
    df_filtered.to_csv(cat_path, index=False)
    print(f"\n  Full catalogue saved : {cat_path}")

    # Earthquake days
    days_path = os.path.join(OUTPUT_DIR,
                             "earthquake_days.csv")
    eq_days = (
        df_filtered
        .groupby("date")
        .agg(
            n_earthquakes  = ("magnitude", "count"),
            max_magnitude  = ("magnitude", "max"),
            min_magnitude  = ("magnitude", "min"),
            mean_depth_km  = ("depth_km",  "mean"),
            matched_paths  = ("matched_paths",
                              lambda x: "|".join(set(
                                  p for paths in x
                                  for p in paths.split(",")
                                  if p
                              )))
        )
        .reset_index()
    )
    eq_days.to_csv(days_path, index=False)
    print(f"  Earthquake days saved: {days_path}")
    print(f"\n  Unique earthquake days : {len(eq_days):,}")
    print("\n Complete")


if __name__ == "__main__":
    main()