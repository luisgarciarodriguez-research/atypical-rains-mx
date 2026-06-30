"""
Mapa de estaciones pluviométricas de México con teselación de Voronoi (k=28).

Genera una figura que superpone tres capas sobre el mapa base de México:
  1. Teselas de Voronoi coloreadas por centroide, recortadas al contorno del país.
  2. Estaciones pluviométricas dispersas por latitud/longitud, coloreadas según
     la etiqueta de clúster ``km_lab`` (k-medias, k=28).
  3. Los 28 centroides marcados con estrella.

Los archivos de entrada son:
  - ``data/raw/nuevas_estaciones.csv.gz``     : TSV con 1,898 estaciones y columna ``km_lab``.
  - ``data/raw/rain_stats_cl_bst_label_group.csv`` : TSV sin encabezado con 28 centroides
    (Long, Lat).
  - ``data/raw/ne_50m_admin_0_countries.zip`` : shapefile Natural Earth 50 m para el
    contorno de México.

La figura se guarda en ``outputs/figures/mapa_voronoi_k28.png`` a 140 dpi.

Proyecto : Detección de Precipitación Atípica en México (2013–2026)
Unidad   : Universidad Nacional Autónoma de México (UNAM)
           Instituto de Investigaciones en Matemáticas Aplicadas y en Sistemas (IIMAS)
           Posgrado en Ciencia e Ingeniería de la Computación (PCIC)
Investigador líder : Dr. José Antonio Neme Castillo
Contribuidor       : Luis García Rodríguez
"""

from __future__ import annotations

import os
import tempfile
import zipfile

import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import Voronoi
from shapely.geometry import MultiPolygon, Polygon

from src.config import DATA_RAW, FIGURES, MEXICO_BBOX

# ── Archivos de entrada ────────────────────────────────────────────────────────
_STATIONS_FILE = DATA_RAW / "nuevas_estaciones.csv.gz"
_CENTROIDS_FILE = DATA_RAW / "rain_stats_cl_bst_label_group.csv"
_COUNTRIES_ZIP = DATA_RAW / "ne_50m_admin_0_countries.zip"

# ── Salida ─────────────────────────────────────────────────────────────────────
OUTPUT_FIG = FIGURES / "mapa_voronoi_k28.png"

# ── Constantes ─────────────────────────────────────────────────────────────────
K = 28
DPI = 600
_BBOX = (
    MEXICO_BBOX["lon_min"],
    MEXICO_BBOX["lat_min"],
    MEXICO_BBOX["lon_max"],
    MEXICO_BBOX["lat_max"],
)


# ── I/O ────────────────────────────────────────────────────────────────────────

def load_stations() -> pd.DataFrame:
    """Carga el conjunto de datos de estaciones y devuelve Long, Lat y km_lab.

    Returns
    -------
    pd.DataFrame
        Columnas: ``Long``, ``Lat``, ``km_lab``.  Filas con NaN eliminadas.
    """
    df = pd.read_csv(_STATIONS_FILE, sep="\t")
    return df[["Long", "Lat", "km_lab"]].dropna()


def load_centroids() -> pd.DataFrame:
    """Carga los 28 centroides k-medias desde el archivo TSV sin encabezado.

    Returns
    -------
    pd.DataFrame
        Columnas: ``Long``, ``Lat``.  Índice: 0..27 (identificador de clúster).
    """
    df = pd.read_csv(_CENTROIDS_FILE, sep="\t", header=None, names=["Long", "Lat"])
    df.index = range(K)
    return df


def load_mexico_geometry() -> tuple[gpd.GeoDataFrame, object]:
    """Extrae la geometría de México del shapefile Natural Earth empaquetado.

    Descomprime el archivo ZIP en un directorio temporal, lee el shapefile con
    GeoPandas y filtra la fila de México por la columna ``SOVEREIGNT``.

    Returns
    -------
    mexico_gdf : gpd.GeoDataFrame
        GeoDataFrame de una sola fila con la geometría de México (EPSG:4326).
    mexico_geom : shapely.geometry
        Unión de todos los polígonos de México; se usa como dominio de recorte.
    """
    with zipfile.ZipFile(_COUNTRIES_ZIP) as zf:
        with tempfile.TemporaryDirectory() as tmpdir:
            zf.extractall(tmpdir)
            shp = next(
                os.path.join(tmpdir, f)
                for f in os.listdir(tmpdir)
                if f.endswith(".shp")
            )
            world = gpd.read_file(shp)

    mexico_gdf = world[world["SOVEREIGNT"] == "Mexico"].copy()
    mexico_gdf = mexico_gdf.set_crs("EPSG:4326", allow_override=True)
    mexico_geom = mexico_gdf.geometry.union_all()
    return mexico_gdf, mexico_geom


# ── Voronoi ────────────────────────────────────────────────────────────────────

def build_voronoi_regions(
    centroids_xy: np.ndarray,
    clip_geom,
) -> list:
    """Construye teselas de Voronoi para los centroides y las recorta a clip_geom.

    Para garantizar que todas las regiones sean finitas antes de la intersección,
    se añaden cuatro puntos auxiliares muy alejados (fuera del bounding box de
    clip_geom).  Los polígonos correspondientes a esos puntos auxiliares se
    descartan; solo se devuelven los 28 polígonos de los centroides originales.

    Parameters
    ----------
    centroids_xy : np.ndarray, shape (n, 2)
        Coordenadas de centroides en orden (longitud, latitud).
    clip_geom : shapely.geometry.base.BaseGeometry
        Geometría de recorte (polígono o multipolígono de México).

    Returns
    -------
    list of shapely.geometry
        Una geometría recortada por centroide.  Puede ser Polygon, MultiPolygon
        o un objeto vacío si el centroide cae completamente fuera de clip_geom.
    """
    minx, miny, maxx, maxy = clip_geom.bounds
    margin = max(maxx - minx, maxy - miny) * 3

    # Puntos auxiliares en las esquinas para cerrar todas las regiones infinitas
    mirror_pts = np.array([
        [minx - margin, miny - margin],
        [maxx + margin, miny - margin],
        [minx - margin, maxy + margin],
        [maxx + margin, maxy + margin],
    ])
    all_pts = np.vstack([centroids_xy, mirror_pts])

    vor = Voronoi(all_pts)
    n_real = len(centroids_xy)

    regions: list = []
    for i in range(n_real):
        region_idx = vor.point_region[i]
        vert_indices = vor.regions[region_idx]

        if -1 in vert_indices or len(vert_indices) == 0:
            # Región todavía infinita (no debería ocurrir con los puntos auxiliares)
            regions.append(clip_geom)
            continue

        poly = Polygon(vor.vertices[vert_indices])
        clipped = poly.intersection(clip_geom)
        regions.append(clipped)

    return regions


# ── Paleta de colores ──────────────────────────────────────────────────────────

def make_color_palette(n: int) -> list:
    """Devuelve n colores cualitativos distintos combinando tab20 y tab20b.

    Toma los 20 colores de ``tab20`` y los primeros ``n-20`` de ``tab20b``
    para formar una paleta de hasta 40 colores sin repetición de tono.

    Parameters
    ----------
    n : int
        Número de colores requeridos (máximo 40).

    Returns
    -------
    list of RGBA tuples
        Lista de longitud ``n``.
    """
    tab20 = list(plt.cm.tab20.colors)
    tab20b = list(plt.cm.tab20b.colors)
    palette = (tab20 + tab20b)[:n]
    return palette


# ── Visualización ──────────────────────────────────────────────────────────────

def _draw_region(ax, geom, color: tuple, alpha: float = 0.30) -> None:
    """Dibuja un polígono (o multipolígono) de Voronoi sobre los ejes dados.

    Rellena el interior con ``color`` a transparencia ``alpha`` y traza el
    contorno con el mismo color a opacidad ligeramente mayor.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Ejes sobre los que se dibuja.
    geom : shapely.geometry.Polygon or MultiPolygon
        Geometría recortada de la tesela de Voronoi.
    color : tuple
        Color RGB o RGBA de la tesela.
    alpha : float
        Transparencia del relleno (0 = transparente, 1 = opaco).
    """
    if geom is None or geom.is_empty:
        return

    polys = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
    for poly in polys:
        if poly.is_empty:
            continue
        x, y = poly.exterior.xy
        ax.fill(x, y, color=color, alpha=alpha, zorder=2)
        ax.plot(x, y, color=color, linewidth=1.0, alpha=0.65, zorder=2)
        # Huecos interiores (islas excluidas dentro del polígono)
        for interior in poly.interiors:
            xi, yi = interior.xy
            ax.fill(xi, yi, color="white", alpha=1.0, zorder=2)


def plot_voronoi_map(
    stations: pd.DataFrame,
    centroids: pd.DataFrame,
    regions: list,
    colors: list,
    mexico_gdf: gpd.GeoDataFrame,
) -> plt.Figure:
    """Genera la figura completa del mapa de Voronoi con estaciones y centroides.

    Capas renderizadas en orden:
      0. Relleno gris claro del contorno de México.
      1. Contorno político de México.
      2. Teselas de Voronoi semitransparentes.
      3. Puntos de estaciones coloreados por ``km_lab``.
      4. Centroides marcados con estrella.
      5. Etiquetas (cXX, n=N) sobre el centroide geométrico de cada celda.

    Parameters
    ----------
    stations : pd.DataFrame
        Columnas: ``Long``, ``Lat``, ``km_lab``.
    centroids : pd.DataFrame
        Columnas: ``Long``, ``Lat``; índice = identificador de clúster (0..K-1).
    regions : list of shapely geometries
        Teselas de Voronoi recortadas, una por centroide.
    colors : list of RGBA tuples
        Paleta de colores, una entrada por clúster.
    mexico_gdf : gpd.GeoDataFrame
        GeoDataFrame con la geometría de México para el mapa base.

    Returns
    -------
    matplotlib.figure.Figure
        Figura lista para guardarse o mostrarse.
    """
    fig, ax = plt.subplots(figsize=(17, 11))

    # — Capa 0: relleno base de México
    mexico_gdf.plot(ax=ax, color="#efefef", edgecolor="none", zorder=0)

    # — Capa 1: contorno de México
    mexico_gdf.boundary.plot(ax=ax, edgecolor="#888888", linewidth=0.9, zorder=1)

    # — Capa 2: teselas de Voronoi
    for k_id, region in enumerate(regions):
        _draw_region(ax, region, color=colors[k_id], alpha=0.28)

    # — Capa 3: estaciones
    cluster_counts = stations["km_lab"].value_counts()
    for k_id in range(K):
        sub = stations[stations["km_lab"] == k_id]
        if sub.empty:
            continue
        ax.scatter(
            sub["Long"], sub["Lat"],
            s=9, color=colors[k_id], alpha=0.80,
            linewidths=0.0, zorder=3,
        )

    # — Capa 4: centroides
    for k_id, row in centroids.iterrows():
        ax.scatter(
            row["Long"], row["Lat"],
            s=160, marker="*",
            color=colors[k_id], edgecolors="#111111",
            linewidths=0.5, zorder=5,
        )

    # — Capa 5: etiquetas sobre el centroide geométrico de cada celda de Voronoi
    for k_id, region in enumerate(regions):
        if region is None or region.is_empty:
            continue
        cx, cy = region.centroid.x, region.centroid.y
        n = cluster_counts.get(k_id, 0)
        ax.text(
            cx, cy,
            f"C{k_id:02d}\nn={n}",
            ha="center", va="center",
            fontsize=11,
            zorder=6,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.55),
        )

    # — Leyenda: 2 columnas fuera del área del mapa
    handles = [
        mpatches.Patch(
            facecolor=colors[k_id],
            edgecolor="#333333",
            linewidth=0.4,
            label=f"C{k_id:02d}  n={cluster_counts.get(k_id, 0)}",
        )
        for k_id in range(K)
    ]
    ax.legend(
        handles=handles,
        ncol=2,
        fontsize=9,
        title="Cluster  (n = stations)",
        title_fontsize=9,
        loc="upper right",
        borderaxespad=0.8,
        framealpha=0.92,
        handlelength=1.4,
    )

    # — Eje y etiquetas
    lon_min, lat_min, lon_max, lat_max = _BBOX
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_xlabel("Longitude", fontsize=10)
    ax.set_ylabel("Latitude", fontsize=10)
    ax.tick_params(labelsize=8)

    ax.set_title(
        "Pluviometric stations of Mexico — k-means classification (k = 28)\n"
        "Voronoi tessellation by centroid",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )

    fig.tight_layout()
    return fig


# ── Punto de entrada ───────────────────────────────────────────────────────────

def main() -> None:
    """Orquesta la carga de datos, el cálculo de Voronoi y el guardado de la figura."""
    print("Cargando estaciones...")
    stations = load_stations()
    print(f"  {len(stations)} estaciones, clústeres: {sorted(stations['km_lab'].unique())[:5]}...")

    print("Cargando centroides...")
    centroids = load_centroids()
    print(f"  {len(centroids)} centroides")

    print("Cargando geometría de México...")
    mexico_gdf, mexico_geom = load_mexico_geometry()

    print("Calculando teselación de Voronoi...")
    xy = centroids[["Long", "Lat"]].values
    regions = build_voronoi_regions(xy, mexico_geom)
    non_empty = sum(1 for r in regions if r is not None and not r.is_empty)
    print(f"  {non_empty}/{K} teselas no vacías")

    print("Generando figura...")
    colors = make_color_palette(K)
    fig = plot_voronoi_map(stations, centroids, regions, colors, mexico_gdf)

    OUTPUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FIG, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Figura guardada en: {OUTPUT_FIG}")


if __name__ == "__main__":
    main()
