# ============================================================
# VLF Monitoring Network Map — using GeoPandas
# JJI Transmitter and 8 UltraMSK Receiver Stations, Japan
# ============================================================

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import numpy as np
from shapely.geometry import Point, LineString
import geodatasets

# ── Station data ─────────────────────────────────────────────
TRANSMITTER = {
    "name": "JJI",
    "label": "JJI Transmitter\n(Ebino, 22.2 kHz)",
    "lat": 32.07,
    "lon": 130.81,
}

STATIONS = {
    "AKT": {"lat": 39.75, "lon": 140.10, "city": "Akita"},
    "ANA": {"lat": 36.24, "lon": 137.98, "city": "Anamizu"},
    "IMZ": {"lat": 34.92, "lon": 136.98, "city": "Imazu"},
    "KMK": {"lat": 43.81, "lon": 144.17, "city": "Kamikawa"},
    "KTU": {"lat": 39.13, "lon": 141.49, "city": "Kamaishi"},
    "NSB": {"lat": 43.57, "lon": 145.60, "city": "Nishibetsu"},
    "STU": {"lat": 33.57, "lon": 131.37, "city": "Shitara"},
    "TYH": {"lat": 34.73, "lon": 138.98, "city": "Toyohashi"},
}

# Label offsets to avoid overlap [dx, dy]
LABEL_OFFSETS = {
    "AKT": (-1.5,  0.3),
    "ANA": (-1.6,  0.1),
    "IMZ": (-1.5, -0.6),
    "KMK": ( 0.3,  0.4),
    "KTU": ( 0.4,  0.2),
    "NSB": ( 0.3,  0.3),
    "STU": (-1.5, -0.6),
    "TYH": ( 0.4, -0.5),
}

# ── Load world map from geodatasets ──────────────────────────
path = geodatasets.get_path("naturalearth.land")
world = gpd.read_file(path)

# Clip to Japan region with a buffer
from shapely.geometry import box
japan_region = box(124, 27, 150, 48)
japan_land = world.clip(japan_region)

# ── Build GeoDataFrames ───────────────────────────────────────
# Transmitter point
tx_gdf = gpd.GeoDataFrame(
    [{"name": TRANSMITTER["name"], "type": "transmitter"}],
    geometry=[Point(TRANSMITTER["lon"], TRANSMITTER["lat"])],
    crs="EPSG:4326"
)

# Receiver points
rx_records = []
rx_geoms = []
for code, info in STATIONS.items():
    rx_records.append({"code": code, "city": info["city"]})
    rx_geoms.append(Point(info["lon"], info["lat"]))
rx_gdf = gpd.GeoDataFrame(rx_records, geometry=rx_geoms, crs="EPSG:4326")

# Signal path lines
path_geoms = []
for info in STATIONS.values():
    path_geoms.append(LineString([
        (TRANSMITTER["lon"], TRANSMITTER["lat"]),
        (info["lon"], info["lat"])
    ]))
paths_gdf = gpd.GeoDataFrame(geometry=path_geoms, crs="EPSG:4326")

# Detection zone circle (~1000 km radius around JJI)
# 1 degree ≈ 111 km → 1000 km ≈ ~9 degrees
# Use buffer on the transmitter point
tx_point = Point(TRANSMITTER["lon"], TRANSMITTER["lat"])
detection_zone = tx_point.buffer(9.5)  # degrees

# ── Plot ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 11))

# Ocean background
ax.set_facecolor("#cde4f0")

# Land
japan_land.plot(ax=ax,
                color="#d6e8b0",
                edgecolor="#888888",
                linewidth=0.7,
                zorder=1)

# Detection zone
zone_gdf = gpd.GeoDataFrame(geometry=[detection_zone], crs="EPSG:4326")
zone_gdf.plot(ax=ax,
              color="none",
              edgecolor="#888888",
              linewidth=1.2,
              linestyle=":",
              zorder=2)

# Signal paths
paths_gdf.plot(ax=ax,
               color="#d4610a",
               linewidth=1.3,
               linestyle="--",
               alpha=0.75,
               zorder=3)

# Receiver stations
rx_gdf.plot(ax=ax,
            color="#1565C0",
            markersize=80,
            marker="o",
            edgecolor="black",
            linewidth=0.8,
            zorder=5)

# Transmitter
tx_gdf.plot(ax=ax,
            color="#cc0000",
            markersize=130,
            marker="^",
            edgecolor="black",
            linewidth=0.8,
            zorder=6)

# ── Annotations ───────────────────────────────────────────────
# Transmitter label
ax.annotate(
    "JJI\n(Ebino)",
    xy=(TRANSMITTER["lon"], TRANSMITTER["lat"]),
    xytext=(TRANSMITTER["lon"] - 2.0, TRANSMITTER["lat"] - 0.8),
    fontsize=9, fontweight="bold", color="#cc0000",
    ha="center", zorder=7,
    arrowprops=dict(arrowstyle="-", color="#cc0000",
                    linewidth=0.8, alpha=0.7)
)

# Receiver labels
for code, info in STATIONS.items():
    dx, dy = LABEL_OFFSETS.get(code, (0.4, 0.2))
    ax.annotate(
        f"{code}\n({info['city']})",
        xy=(info["lon"], info["lat"]),
        xytext=(info["lon"] + dx, info["lat"] + dy),
        fontsize=8,
        color="#0d2f6e",
        ha="center",
        zorder=7,
        bbox=dict(boxstyle="round,pad=0.2",
                  facecolor="white",
                  edgecolor="none",
                  alpha=0.7)
    )

# Detection zone label
ax.annotate(
    "Detection Zone\n(~1000 km radius)",
    xy=(TRANSMITTER["lon"] + 9.5, TRANSMITTER["lat"]),
    xytext=(TRANSMITTER["lon"] + 9.2, TRANSMITTER["lat"] - 3.5),
    fontsize=8,
    color="#555555",
    ha="center",
    arrowprops=dict(arrowstyle="->", color="#888888", linewidth=0.8)
)

# ── Map formatting ────────────────────────────────────────────
ax.set_xlim(126, 149)
ax.set_ylim(29.5, 46.5)
ax.set_xlabel("Longitude (°E)", fontsize=10)
ax.set_ylabel("Latitude (°N)", fontsize=10)
ax.set_title(
    "VLF Monitoring Network — JJI Transmitter and\n"
    "UltraMSK Receiver Stations, Japan (2014–2017)",
    fontsize=13, fontweight="bold", pad=14
)

# Gridlines
ax.grid(True, color="white", linewidth=0.5, alpha=0.8, zorder=0)
ax.set_xticks(np.arange(126, 150, 2))
ax.set_yticks(np.arange(30, 47, 2))
ax.tick_params(labelsize=9)

# ── Legend ────────────────────────────────────────────────────
legend_elements = [
    mlines.Line2D([0], [0], marker="^", color="w",
                  markerfacecolor="#cc0000", markeredgecolor="black",
                  markersize=13,
                  label="JJI Transmitter (Ebino, 22.2 kHz)"),
    mlines.Line2D([0], [0], marker="o", color="w",
                  markerfacecolor="#1565C0", markeredgecolor="black",
                  markersize=10,
                  label="UltraMSK Receiver Stations (×8)"),
    mlines.Line2D([0], [0], color="#d4610a", linewidth=1.5,
                  linestyle="--",
                  label="Transmitter-Receiver Signal Paths"),
    mlines.Line2D([0], [0], color="#888888", linewidth=1.2,
                  linestyle=":",
                  label="Approximate Detection Zone (~1000 km)"),
]
ax.legend(handles=legend_elements,
          loc="lower right",
          fontsize=9,
          framealpha=0.92,
          edgecolor="#aaaaaa",
          borderpad=0.8)

import os
os.makedirs("outputs", exist_ok=True)
plt.subplots_adjust(left=0.08, right=0.97, top=0.95, bottom=0.07)
plt.savefig("outputs/vlf_network_map.png", dpi=200, bbox_inches="tight")
plt.show()
print("Map saved to outputs/vlf_network_map.png")