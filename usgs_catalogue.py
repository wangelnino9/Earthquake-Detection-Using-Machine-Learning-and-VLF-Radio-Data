"""
Step 3 — Load USGS Earthquake Catalogue with Fresnel Zone Ellipse Filtering
============================================================================
Downloads USGS NEIC earthquake catalogue for Japan region (2014-2017),
filters for M >= 3.0, and keeps only earthquakes that fall within the
first Fresnel zone ellipse of at least one JJI transmitter-receiver path.

Fresnel Zone Ellipse:
- T (transmitter JJI) and R (receiver station) are the two foci
- For any point P inside the ellipse: TP + PR <= TR + lambda/2
- lambda = c/f = 3e8 / 22200 = 13,514 m (JJI at 22.2 kHz)
- Semi-major axis: a = c + lambda/4  (where c = TR/2)
- Semi-minor axis: b = sqrt(a^2 - c^2)
- Buffer factor: 1.5x first Fresnel zone (recommended for earthquake studies)
"""

import requests
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# ── Configuration ─────────────────────────────────────────────

# JJI Transmitter (Ebino, Japan)
JJI_LAT = 32.08
JJI_LON = 130.82
JJI_FREQ = 22200  # Hz

# Speed of light
C_LIGHT = 3e8  # m/s

# Fresnel zone buffer multiplier
# 1.0 = strict first Fresnel zone
# 1.5 = recommended buffer for earthquake studies
FRESNEL_BUFFER = 1.5

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

# USGS query parameters
START_DATE  = "2014-01-01"
END_DATE    = "2017-02-28"
MIN_MAG     = 3.0

# Broad bounding box for initial USGS query (Japan region)
# Will be further filtered by Fresnel zone ellipse
MIN_LAT = 25.0
MAX_LAT = 48.0
MIN_LON = 125.0
MAX_LON = 150.0

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Helper Functions ──────────────────────────────────────────

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate great circle distance between two points (metres).
    """
    R = 6371000  # Earth radius in metres
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def compute_fresnel_params(tx_lat, tx_lon, rx_lat, rx_lon, freq, buffer=1.5):
    """
    Compute Fresnel zone ellipse parameters for one transmitter-receiver path.

    Returns dict with:
        TR      : transmitter-receiver distance (m)
        c       : half focal distance (m)
        lam     : wavelength (m)
        a       : semi-major axis with buffer (m)
        b       : semi-minor axis with buffer (m)
        centre  : midpoint (lat, lon)
        bearing : bearing from TX to RX (degrees)
    """
    lam = C_LIGHT / freq                        # wavelength in metres
    TR  = haversine(tx_lat, tx_lon, rx_lat, rx_lon)  # full path distance
    c   = TR / 2                                # half focal distance

    # First Fresnel zone
    a_strict = c + lam / 4
    b_strict = np.sqrt(max(a_strict**2 - c**2, 0))

    # Apply buffer
    a = a_strict * buffer
    b = b_strict * buffer

    # Centre of ellipse (midpoint of TX-RX path)
    centre_lat = (tx_lat + rx_lat) / 2
    centre_lon = (tx_lon + rx_lon) / 2

    # Bearing from TX to RX
    y = np.sin(np.radians(rx_lon - tx_lon)) * np.cos(np.radians(rx_lat))
    x = (np.cos(np.radians(tx_lat)) * np.sin(np.radians(rx_lat)) -
         np.sin(np.radians(tx_lat)) * np.cos(np.radians(rx_lat)) *
         np.cos(np.radians(rx_lon - tx_lon)))
    bearing = np.degrees(np.arctan2(y, x)) % 360

    return {
        "TR":      TR,
        "c":       c,
        "lam":     lam,
        "a":       a,
        "b":       b,
        "centre":  (centre_lat, centre_lon),
        "bearing": bearing,
    }


def point_in_fresnel_ellipse(eq_lat, eq_lon, tx_lat, tx_lon, rx_lat, rx_lon, params):
    """
    Check whether an earthquake point falls inside the Fresnel zone ellipse.

    Uses the fundamental ellipse property:
        distance(P, F1) + distance(P, F2) <= 2a

    Where F1 = transmitter, F2 = receiver, P = earthquake location.
    This is exact regardless of ellipse orientation.

    Returns True if inside ellipse, False otherwise.
    """
    d_tx = haversine(eq_lat, eq_lon, tx_lat, tx_lon)
    d_rx = haversine(eq_lat, eq_lon, rx_lat, rx_lon)
    return (d_tx + d_rx) <= (2 * params["a"])


# ── Step 1: Download USGS Catalogue ──────────────────────────

def download_usgs_catalogue():
    """
    Download earthquake catalogue from USGS ComCat API.
    Returns raw DataFrame.
    """
    print("Downloading USGS earthquake catalogue...")
    print(f"  Region : {MIN_LAT}°N–{MAX_LAT}°N, {MIN_LON}°E–{MAX_LON}°E")
    print(f"  Period : {START_DATE} to {END_DATE}")
    print(f"  Min Mag: M ≥ {MIN_MAG}")

    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    params = {
        "format":     "csv",
        "starttime":  START_DATE,
        "endtime":    END_DATE,
        "minmagnitude": MIN_MAG,
        "minlatitude":  MIN_LAT,
        "maxlatitude":  MAX_LAT,
        "minlongitude": MIN_LON,
        "maxlongitude": MAX_LON,
        "orderby":    "time",
    }

    try:
        response = requests.get(url, params=params, timeout=120)
        response.raise_for_status()
        from io import StringIO
        df = pd.read_csv(StringIO(response.text))
        print(f"  Downloaded {len(df):,} earthquakes from USGS")
        return df
    except requests.exceptions.RequestException as e:
        print(f"  ERROR downloading catalogue: {e}")
        print("  Tip: Check your internet connection and try again.")
        return None


# ── Step 2: Clean and Prepare Catalogue ──────────────────────

def clean_catalogue(df):
    """
    Clean USGS catalogue — select relevant columns, parse dates.
    """
    # USGS CSV columns we need
    cols_needed = ["time", "latitude", "longitude", "depth", "mag", "magType",
                   "place", "type"]
    cols_available = [c for c in cols_needed if c in df.columns]
    df = df[cols_available].copy()

    # Keep only earthquakes (not explosions, quarry blasts etc.)
    if "type" in df.columns:
        df = df[df["type"] == "earthquake"].copy()

    # Parse datetime
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df["date"] = df["time"].dt.date

    # Drop rows missing key fields
    df = df.dropna(subset=["latitude", "longitude", "mag"])
    df = df.rename(columns={"latitude": "lat", "longitude": "lon",
                             "mag": "magnitude", "depth": "depth_km"})

    print(f"  After cleaning: {len(df):,} earthquake events")
    return df


# ── Step 3: Compute Fresnel Zones for All Paths ───────────────

def compute_all_fresnel_zones():
    """
    Compute Fresnel zone parameters for all 8 JJI-station paths.
    """
    print("\nComputing Fresnel zone ellipses...")
    print(f"  JJI transmitter : ({JJI_LAT}°N, {JJI_LON}°E)")
    print(f"  Frequency       : {JJI_FREQ/1000} kHz")
    print(f"  Wavelength      : {C_LIGHT/JJI_FREQ/1000:.1f} km")
    print(f"  Buffer factor   : {FRESNEL_BUFFER}×")
    print()

    zones = {}
    print(f"  {'Station':<8} {'TR (km)':>10} {'a (km)':>10} "
          f"{'b (km)':>10} {'b_strict (km)':>14}")
    print("  " + "-"*58)

    for stn, (rx_lat, rx_lon) in STATIONS.items():
        p = compute_fresnel_params(JJI_LAT, JJI_LON,
                                   rx_lat, rx_lon,
                                   JJI_FREQ, FRESNEL_BUFFER)
        zones[stn] = p

        # Also compute strict (no buffer) b for display
        a_s = p["c"] + p["lam"] / 4
        b_s = np.sqrt(max(a_s**2 - p["c"]**2, 0))

        print(f"  {stn:<8} {p['TR']/1000:>10.1f} {p['a']/1000:>10.1f} "
              f"{p['b']/1000:>10.1f} {b_s/1000:>14.1f}")

    return zones


# ── Step 4: Filter Earthquakes by Fresnel Zone ────────────────

def filter_by_fresnel_zone(df, zones):
    """
    Keep earthquakes that fall inside at least one Fresnel zone ellipse.

    For each earthquake we check all 8 paths using the exact ellipse property:
        d(P, TX) + d(P, RX) <= 2a

    Adds columns:
        in_fresnel      : True if inside any Fresnel ellipse
        fresnel_paths   : list of stations whose Fresnel zone contains this EQ
    """
    print("\nFiltering earthquakes by Fresnel zone ellipses...")

    in_fresnel  = []
    path_list   = []

    for _, row in df.iterrows():
        eq_lat = row["lat"]
        eq_lon = row["lon"]
        matched = []

        for stn, (rx_lat, rx_lon) in STATIONS.items():
            params = zones[stn]
            if point_in_fresnel_ellipse(eq_lat, eq_lon,
                                        JJI_LAT, JJI_LON,
                                        rx_lat, rx_lon,
                                        params):
                matched.append(stn)

        in_fresnel.append(len(matched) > 0)
        path_list.append(",".join(matched) if matched else "")

    df = df.copy()
    df["in_fresnel"]    = in_fresnel
    df["fresnel_paths"] = path_list

    filtered = df[df["in_fresnel"]].copy()
    print(f"  Before filtering : {len(df):,} earthquakes")
    print(f"  Inside Fresnel   : {len(filtered):,} earthquakes")
    print(f"  Excluded         : {len(df) - len(filtered):,} earthquakes")

    return filtered


# ── Step 5: Summary Statistics ────────────────────────────────

def print_summary(df):
    """
    Print summary statistics of filtered catalogue.
    """
    print("\n" + "="*60)
    print("FILTERED CATALOGUE SUMMARY")
    print("="*60)
    print(f"  Total earthquakes     : {len(df):,}")
    print(f"  Date range            : {df['date'].min()} to {df['date'].max()}")
    print(f"  Unique earthquake days: {df['date'].nunique():,}")
    print(f"  Magnitude range       : {df['magnitude'].min():.1f} – "
          f"{df['magnitude'].max():.1f}")
    print(f"  Mean magnitude        : {df['magnitude'].mean():.2f}")
    print()

    # Magnitude distribution
    bins = [3.0, 4.0, 5.0, 6.0, 7.0, 10.0]
    labels = ["M3.0–3.9", "M4.0–4.9", "M5.0–5.9", "M6.0–6.9", "M7.0+"]
    df["mag_bin"] = pd.cut(df["magnitude"], bins=bins, labels=labels,
                           right=False)
    print("  Magnitude distribution:")
    for label, count in df["mag_bin"].value_counts().sort_index().items():
        bar = "█" * (count // 10)
        print(f"    {label} : {count:>5,}  {bar}")

    print()
    print("  Earthquakes per year:")
    df["year"] = pd.to_datetime(df["date"].astype(str)).dt.year
    for year, count in df.groupby("year").size().items():
        print(f"    {year} : {count:,}")


# ── Step 6: Visualise ─────────────────────────────────────────

# def visualise(df, zones):
    """
    Plot earthquake locations with Fresnel zone ellipses overlaid on a map.
    """
    print("\nGenerating visualisation...")

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.patch.set_facecolor("white")

    colors = {
        "AKT": "#2E6DA4", "ANA": "#2E7D52", "IMZ": "#C8922A",
        "KMK": "#1C7293", "KTU": "#B45309", "NSB": "#7B2D8B",
        "STU": "#C0392B", "TYH": "#5B8DB8"
    }

    for ax_idx, ax in enumerate(axes):
        ax.set_facecolor("#F0F4F8")

        # Plot Fresnel zone ellipses (approximate as scatter of points)
        for stn, (rx_lat, rx_lon) in STATIONS.items():
            p = zones[stn]
            # Sample points on the ellipse boundary using the ellipse property
            # Generate points that satisfy d(P,TX) + d(P,RX) = 2a
            # We parametrize using bearing from centre
            centre_lat, centre_lon = p["centre"]
            a_deg = p["a"] / 111000  # approximate degrees
            b_deg = p["b"] / 111000

            # Bearing of major axis
            bearing_rad = np.radians(p["bearing"])

            theta = np.linspace(0, 2 * np.pi, 200)
            # Ellipse in local frame
            x_local = a_deg * np.cos(theta)
            y_local = b_deg * np.sin(theta)
            # Rotate by bearing
            x_rot = (x_local * np.cos(bearing_rad) -
                     y_local * np.sin(bearing_rad))
            y_rot = (x_local * np.sin(bearing_rad) +
                     y_local * np.cos(bearing_rad))

            ell_lon = centre_lon + x_rot / np.cos(np.radians(centre_lat))
            ell_lat = centre_lat + y_rot

            ax.plot(ell_lon, ell_lat,
                    color=colors[stn], linewidth=1.2,
                    alpha=0.6, linestyle="--",
                    label=f"{stn} zone" if ax_idx == 0 else "")

            # Path line
            ax.plot([JJI_LON, rx_lon], [JJI_LAT, rx_lat],
                    color=colors[stn], linewidth=0.8, alpha=0.5)

        # Plot earthquakes
        if ax_idx == 0:
            # All earthquakes before Fresnel filtering
            label = "All M≥3.0 Japan EQs"
            eq_plot = df
            title = f"Before Fresnel Filtering\n({len(df):,} earthquakes)"
        else:
            label = "Inside Fresnel zone"
            eq_plot = df[df["in_fresnel"]]
            title = f"After Fresnel Filtering\n({len(eq_plot):,} earthquakes)"

        sc = ax.scatter(eq_plot["lon"], eq_plot["lat"],
                        c=eq_plot["magnitude"],
                        cmap="YlOrRd", s=8, alpha=0.5,
                        vmin=3.0, vmax=7.5,
                        zorder=3, label=label)

        # JJI transmitter
        ax.scatter(JJI_LON, JJI_LAT, marker="^", s=200,
                   color="#1B3A5C", zorder=5, label="JJI Transmitter")
        ax.annotate("JJI", (JJI_LON, JJI_LAT),
                    textcoords="offset points", xytext=(6, 4),
                    fontsize=8, color="#1B3A5C", fontweight="bold")

        # Receiver stations
        for stn, (lat, lon) in STATIONS.items():
            ax.scatter(lon, lat, marker="v", s=80,
                       color=colors[stn], zorder=5)
            ax.annotate(stn, (lon, lat),
                        textcoords="offset points", xytext=(4, 3),
                        fontsize=7, color=colors[stn], fontweight="bold")

        ax.set_xlim(125, 150)
        ax.set_ylim(25, 48)
        ax.set_xlabel("Longitude (°E)", fontsize=10)
        ax.set_ylabel("Latitude (°N)", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold", color="#1B3A5C")
        ax.grid(True, alpha=0.3, color="white")
        ax.tick_params(labelsize=8)

        if ax_idx == 1:
            plt.colorbar(sc, ax=ax, label="Magnitude", shrink=0.8)

    plt.suptitle(
        "USGS Earthquake Catalogue — Fresnel Zone Filtering\n"
        f"JJI Transmitter (22.2 kHz) | M≥{MIN_MAG} | {START_DATE} to {END_DATE}",
        fontsize=12, fontweight="bold", color="#1B3A5C", y=1.01
    )
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "step3_fresnel_map.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Map saved to: {out_path}")


# ── Main ──────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("STEP 3 — USGS EARTHQUAKE CATALOGUE (FRESNEL ZONE FILTER)")
    print("=" * 60)

    # 1. Compute Fresnel zones
    zones = compute_all_fresnel_zones()

    # 2. Download USGS catalogue
    df_raw = download_usgs_catalogue()
    if df_raw is None:
        return

    # 3. Clean catalogue
    df_clean = clean_catalogue(df_raw)

    # 4. Filter by Fresnel zone
    df_filtered = filter_by_fresnel_zone(df_clean, zones)

    # 5. Summary statistics
    # print_summary(df_filtered)

    # 6. Visualise
    # visualise(df_clean, zones)

    # 7. Save outputs
    # Full filtered catalogue
    cat_path = os.path.join(OUTPUT_DIR, "usgs_catalogue_fresnel.csv")
    df_filtered.to_csv(cat_path, index=False)
    print(f"\n  Catalogue saved to : {cat_path}")

    # Unique earthquake days only (for label creation in Step 4)
    days_path = os.path.join(OUTPUT_DIR, "earthquake_days.csv")
    eq_days = (df_filtered.groupby("date")
               .agg(
                   n_earthquakes=("magnitude", "count"),
                   max_magnitude=("magnitude", "max"),
                   min_magnitude=("magnitude", "min"),
                   mean_depth=("depth_km", "mean"),
                   paths=("fresnel_paths", lambda x: "|".join(set(
                       p for paths in x for p in paths.split(",") if p)))
               )
               .reset_index())
    eq_days.to_csv(days_path, index=False)
    print(f"  Earthquake days saved to: {days_path}")
    print(f"\n  Unique earthquake days: {len(eq_days):,}")

    print("\n✅ Step 3 complete.")
    print("   Next: Step 4 — Label creation")


if __name__ == "__main__":
    main()