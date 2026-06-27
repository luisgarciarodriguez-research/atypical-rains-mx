"""
Visualización geoespacial de la red pluviométrica nacional — Tarea T1.4.

Genera tres mapas sobre el territorio mexicano usando el shapefile Natural Earth
(50 m) como fondo cartográfico, proyección Web Mercator (EPSG:3857) con basemap
de CartoDB Positron:

  T1.4.1  Mapa de cobertura: scatter de 1,959 estaciones coloreado por
          pct_complete (verde = completa, rojo = escasa).
  T1.4.2  Mapa de kriging ordinario (modelo esférico de T1.3.5) de la
          precipitación anual media interpolada a una cuadrícula 180 × 120.
          Los puntos fuera del polígono de México se enmascaran como NaN.
  T1.4.3  Cuatro mapas trimestrales (DEF, MAM, JJA, SON) con mediana
          estacional por estación, escala de color compartida para
          comparabilidad directa entre estaciones.

Dependencia de datos
--------------------
  Archivo  : data/raw/ne_50m_admin_0_countries.zip
  Fuente   : Natural Earth Admin 0 Countries, resolución 50 m (v5.x)
             https://naciscdn.org/naturalearth/50m/cultural/ne_50m_admin_0_countries.zip
  Motivo   : La resolución 50 m ofrece suficiente detalle para trazar el
             polígono de México y enmascarar puntos de kriging fuera del
             territorio nacional, sin el peso del archivo 10 m (~60 MB).
  Obtención: El archivo está incluido en data/raw/ del repositorio. Si se
             necesita volver a obtenerlo, ``_download_shapefile()`` lo descarga
             automáticamente desde la URL anterior al primer uso.

Punto de entrada: ``run_t1_4(verbose=True)``.

Proyecto : Detección de Precipitación Atípica en México (2013–2026)
Unidad   : Universidad Nacional Autónoma de México (UNAM)
           Instituto de Investigaciones en Matemáticas Aplicadas y en Sistemas (IIMAS)
           Posgrado en Ciencia e Ingeniería de la Computación (PCIC)
Investigador líder : Dr. José Antonio Neme Castillo
Contribuidor       : Luis García Rodríguez
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.colorbar import ColorbarBase
import geopandas as gpd
import contextily as cx
from pykrige.ok import OrdinaryKriging
from pathlib import Path
from shapely.geometry import Point

from src.config import (
    DATA_RAW, DATA_PROCESSED, DATA_CATALOGS, FIGURES, MEXICO_BBOX, SEED,
)
from src.loading import parse_rain_columns

# ── Constantes ────────────────────────────────────────────────────────────────

_NE_110M_URL = (
    "https://naciscdn.org/naturalearth/110m/cultural/"
    "ne_110m_admin_0_countries.zip"
)
_NE_50M_URL = (
    "https://naciscdn.org/naturalearth/50m/cultural/"
    "ne_50m_admin_0_countries.zip"
)
_LOCAL_110M = DATA_CATALOGS / "ne_110m_admin_0_countries.zip"
_LOCAL_50M  = DATA_RAW / "ne_50m_admin_0_countries.zip"

SEASONS = {
    "DEF": ([12, 1, 2],  "Invierno (Dic–Feb)"),
    "MAM": ([3, 4, 5],   "Primavera (Mar–May)"),
    "JJA": ([6, 7, 8],   "Verano (Jun–Ago)"),
    "SON": ([9, 10, 11], "Otoño (Sep–Nov)"),
}

# Parámetros del variograma esférico de T1.3.5
_VARIO_PARAMS = {"psill": 6719.85, "range": 31.61, "nugget": 1440.91}


# ── Utilidades ────────────────────────────────────────────────────────────────

def _sorted_rain_cols(rain_col_map: dict) -> list[str]:
    return sorted(rain_col_map, key=lambda c: (rain_col_map[c][1], rain_col_map[c][0]))


def _download_shapefile(url: str, local: Path) -> Path:
    """Descarga y cachea un shapefile ZIP si no existe localmente."""
    if local.exists():
        return local
    import urllib.request
    DATA_CATALOGS.mkdir(parents=True, exist_ok=True)
    print(f"    Descargando {url.split('/')[-1]}...")
    urllib.request.urlretrieve(url, local)
    return local


def _load_world(resolution: str = "110m") -> gpd.GeoDataFrame:
    """Carga el shapefile de países Natural Earth (110m o 50m)."""
    url   = _NE_50M_URL  if resolution == "50m" else _NE_110M_URL
    local = _LOCAL_50M   if resolution == "50m" else _LOCAL_110M
    path = _download_shapefile(url, local)
    world = gpd.read_file(path)
    # Columna de nombre del país (varía según versión NE)
    name_col = next(
        (c for c in world.columns if c.upper() in ("NAME", "SOVEREIGNT", "ADMIN")),
        world.columns[0],
    )
    world = world.rename(columns={name_col: "_name"})
    return world


def _mexico_gdf(resolution: str = "50m") -> gpd.GeoDataFrame:
    """Devuelve el polígono de México en EPSG:4326."""
    world = _load_world(resolution)
    mex = world[world["_name"].str.strip() == "Mexico"].copy()
    if mex.empty:
        # Fallback: buscar por bounding box aproximado
        mex = world.cx[-118.5:-86.5, 14.5:33.0].copy()
    return mex.to_crs(epsg=4326)


def _stations_gdf(df: pd.DataFrame) -> gpd.GeoDataFrame:
    """Convierte el DataFrame de estaciones a GeoDataFrame EPSG:4326."""
    mask = df["Long"].notna() & df["Lat"].notna()
    return gpd.GeoDataFrame(
        df[mask].copy(),
        geometry=gpd.points_from_xy(df.loc[mask, "Long"], df.loc[mask, "Lat"]),
        crs="EPSG:4326",
    )


def _add_basemap(ax, zoom: int = 5) -> None:
    """Añade basemap de CartoDB Positron; silencia errores de red."""
    try:
        cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, zoom=zoom, alpha=0.6)
    except Exception:
        try:
            cx.add_basemap(ax, source=cx.providers.OpenStreetMap.Mapnik, zoom=zoom, alpha=0.5)
        except Exception:
            pass  # sin tiles, el mapa sigue siendo válido


def _bbox_extent(pad: float = 0.5) -> tuple[float, float, float, float]:
    bb = MEXICO_BBOX
    return (
        bb["lon_min"] - pad, bb["lon_max"] + pad,
        bb["lat_min"] - pad, bb["lat_max"] + pad,
    )


def _web_extent() -> tuple[float, float, float, float]:
    """Bounding box en EPSG:3857 (metros) con padding."""
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    bb = MEXICO_BBOX
    xmin, ymin = tr.transform(bb["lon_min"] - 0.5, bb["lat_min"] - 0.5)
    xmax, ymax = tr.transform(bb["lon_max"] + 0.5, bb["lat_max"] + 0.5)
    return xmin, xmax, ymin, ymax


# ── T1.4.1 — Mapa de cobertura ────────────────────────────────────────────────

def t1_4_1_coverage_map(df: pd.DataFrame, verbose: bool = True) -> Path:
    """
    Scatter de 1,959 estaciones coloreado por pct_complete,
    con contorno de México y basemap topográfico.
    """
    if verbose:
        print("    Cargando shapefile de México...")
    mexico = _mexico_gdf("50m")
    mexico_web = mexico.to_crs(epsg=3857)

    gdf = _stations_gdf(df)
    gdf_web = gdf.to_crs(epsg=3857)

    fig, ax = plt.subplots(figsize=(14, 10))

    # Fondo: polígono de México
    mexico_web.plot(ax=ax, color="whitesmoke", edgecolor="#444", linewidth=0.8, alpha=0.5)

    # Estaciones coloreadas por pct_complete
    scatter = gdf_web.plot(
        ax=ax,
        column="pct_complete",
        cmap="RdYlGn",
        markersize=7,
        alpha=0.85,
        legend=True,
        legend_kwds={
            "label": "% Completitud",
            "orientation": "vertical",
            "shrink": 0.6,
            "pad": 0.02,
        },
    )

    _add_basemap(ax, zoom=5)

    xmin, xmax, ymin, ymax = _web_extent()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_axis_off()
    ax.set_title(
        "T1.4.1 — Cobertura de estaciones pluviométricas\n"
        f"n={len(gdf):,} · coloreado por % completitud (verde = completa)",
        fontsize=12,
    )

    out = FIGURES / "map_coverage.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    if verbose:
        print(f"    → {out.name}")
    return out


# ── T1.4.2 — Mapa de kriging ─────────────────────────────────────────────────

def t1_4_2_kriging_map(
    df: pd.DataFrame, rain_cols: list, verbose: bool = True
) -> Path:
    """
    Precipitación anual media interpolada con kriging ordinario
    usando el modelo esférico ajustado en T1.3.5.
    """
    mask_complete = df["pct_complete"] >= 0.80
    annual_mean = df[rain_cols].mean(axis=1)

    lons = df.loc[mask_complete, "Long"].values
    lats = df.loc[mask_complete, "Lat"].values
    zvals = annual_mean[mask_complete].values

    valid = ~(np.isnan(lons) | np.isnan(lats) | np.isnan(zvals))
    lons, lats, zvals = lons[valid], lats[valid], zvals[valid]

    if verbose:
        print(f"    Kriging con {len(lons)} estaciones (pct_complete ≥ 80%)...")

    # Ajustar kriging con parámetros pre-fijados del variograma de T1.3.5
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        OK = OrdinaryKriging(
            lons, lats, zvals,
            variogram_model="spherical",
            variogram_parameters=_VARIO_PARAMS,
            verbose=False,
            enable_plotting=False,
        )

    # Grid de predicción sobre México
    bb = MEXICO_BBOX
    xi = np.linspace(bb["lon_min"], bb["lon_max"], 180)
    yi = np.linspace(bb["lat_min"], bb["lat_max"], 120)

    if verbose:
        print(f"    Prediciendo en grid {len(xi)}×{len(yi)} = {len(xi)*len(yi):,} pts...")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        z_pred, z_var = OK.execute("grid", xi, yi)

    # Máscara: puntos fuera de México → NaN
    mexico = _mexico_gdf("50m")
    mexico_union = mexico.geometry.union_all()

    lon_grid, lat_grid = np.meshgrid(xi, yi)
    mask_land = np.array([
        [mexico_union.contains(Point(x, y)) for x in xi]
        for y in yi
    ], dtype=bool)
    z_masked = np.where(mask_land, z_pred.data, np.nan)

    # Plot
    mexico_web = mexico.to_crs(epsg=3857)
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    xi_web, _ = tr.transform(xi, np.zeros_like(xi))
    _, yi_web = tr.transform(np.zeros_like(yi), yi)
    lon_web, lat_web = np.meshgrid(xi_web, yi_web)

    fig, ax = plt.subplots(figsize=(14, 10))
    mexico_web.plot(ax=ax, color="none", edgecolor="#333", linewidth=0.8)

    pcm = ax.pcolormesh(
        lon_web, lat_web, z_masked,
        cmap="YlOrRd", shading="auto",
        vmin=0, vmax=np.nanpercentile(z_masked, 97),
    )
    plt.colorbar(pcm, ax=ax, label="Precipitación anual media (mm)", shrink=0.65)

    # Puntos de estaciones encima
    gdf = _stations_gdf(df[mask_complete]).to_crs(epsg=3857)
    gdf.plot(ax=ax, color="black", markersize=4, alpha=0.5, zorder=3)

    _add_basemap(ax, zoom=5)
    xmin, xmax, ymin, ymax = _web_extent()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_axis_off()
    ax.set_title(
        "T1.4.2 — Precipitación anual media interpolada (Kriging Ordinario, modelo esférico)\n"
        f"n={len(lons)} estaciones · pct_complete ≥ 80% · "
        f"psill={_VARIO_PARAMS['psill']:.0f} mm²  range={_VARIO_PARAMS['range']:.1f}°",
        fontsize=11,
    )

    out = FIGURES / "map_kriging.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    if verbose:
        print(f"    → {out.name}")
    return out


# ── T1.4.3 — Mapas trimestrales ──────────────────────────────────────────────

def t1_4_3_seasonal_maps(
    df: pd.DataFrame,
    rain_cols: list,
    rain_col_map: dict,
    verbose: bool = True,
) -> Path:
    """
    Cuatro mapas trimestrales (DEF, MAM, JJA, SON):
    mediana de precipitación por estación en los meses del trimestre,
    agregada sobre todos los años disponibles.
    """
    mexico = _mexico_gdf("50m")
    mexico_web = mexico.to_crs(epsg=3857)

    gdf_base = _stations_gdf(df)
    gdf_web = gdf_base.to_crs(epsg=3857)

    # Calcular mediana estacional por estación
    for season, (months, _) in SEASONS.items():
        season_cols = [c for c, (m, _y) in rain_col_map.items() if m in months]
        gdf_web[f"median_{season}"] = df.loc[
            df["Long"].notna(), season_cols
        ].median(axis=1).values

    # Escala de color global para comparabilidad entre estaciones
    all_vals = np.concatenate([
        gdf_web[f"median_{s}"].dropna().values for s in SEASONS
    ])
    vmin, vmax = 0, float(np.percentile(all_vals, 97))

    fig, axes = plt.subplots(2, 2, figsize=(18, 13))
    axes_flat = axes.flatten()

    cmap = plt.get_cmap("YlOrRd")
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    for ax, (season, (months, label)) in zip(axes_flat, SEASONS.items()):
        col = f"median_{season}"
        valid = gdf_web[col].notna()
        n_valid = valid.sum()

        # Fondo: polígono México
        mexico_web.plot(ax=ax, color="whitesmoke", edgecolor="#444",
                        linewidth=0.6, alpha=0.6)

        # Estaciones con dato
        sc = ax.scatter(
            gdf_web.loc[valid, "geometry"].x,
            gdf_web.loc[valid, "geometry"].y,
            c=gdf_web.loc[valid, col].values,
            cmap=cmap, norm=norm,
            s=12, alpha=0.9, zorder=3,
        )

        # Estaciones sin dato: marcador gris
        no_data = ~valid
        if no_data.sum() > 0:
            ax.scatter(
                gdf_web.loc[no_data, "geometry"].x,
                gdf_web.loc[no_data, "geometry"].y,
                c="lightgray", s=5, alpha=0.4, zorder=2,
            )

        _add_basemap(ax, zoom=5)
        xmin, xmax, ymin_, ymax_ = _web_extent()
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin_, ymax_)
        ax.set_axis_off()

        season_median = gdf_web.loc[valid, col].median()
        ax.set_title(
            f"{label}\n"
            f"Mediana nacional={season_median:.1f} mm  n={n_valid:,}",
            fontsize=10,
        )

    # Colorbar compartida
    fig.subplots_adjust(right=0.88, hspace=0.06, wspace=0.02)
    cax = fig.add_axes([0.90, 0.15, 0.025, 0.70])
    cb = ColorbarBase(cax, cmap=cmap, norm=norm, orientation="vertical")
    cb.set_label("Mediana precipitación (mm)", fontsize=10)

    fig.suptitle(
        "T1.4.3 — Medianas de precipitación por trimestre\n"
        "(escala de color común entre estaciones · gris = sin dato)",
        fontsize=12, y=0.99,
    )

    out = FIGURES / "map_seasonal.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    if verbose:
        print(f"    → {out.name}")
    return out


# ── Punto de entrada ──────────────────────────────────────────────────────────

def run_t1_4(verbose: bool = True) -> None:
    """Ejecuta T1.4 completo: tres mapas geoespaciales."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    DATA_CATALOGS.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("=" * 60)
        print("T1.4 — Visualización Geoespacial")
        print("=" * 60)

    df = pd.read_parquet(DATA_PROCESSED / "lluvia_clean.parquet")
    rain_col_map = parse_rain_columns(df.columns)
    rain_cols = _sorted_rain_cols(rain_col_map)

    if verbose:
        print(f"Datos: {df.shape[0]:,} estaciones × {len(rain_cols)} meses\n")

    # T1.4.1
    if verbose:
        print("[T1.4.1] Mapa de cobertura...")
    t1_4_1_coverage_map(df, verbose=verbose)

    # T1.4.2
    if verbose:
        print("\n[T1.4.2] Mapa de kriging (precipitación anual media)...")
    t1_4_2_kriging_map(df, rain_cols, verbose=verbose)

    # T1.4.3
    if verbose:
        print("\n[T1.4.3] Mapas trimestrales (DEF, MAM, JJA, SON)...")
    t1_4_3_seasonal_maps(df, rain_cols, rain_col_map, verbose=verbose)

    if verbose:
        print("\n[OK] T1.4 completado.")
        print("\nFiguras generadas:")
        for f in sorted(FIGURES.glob("map_*.png")):
            size_kb = f.stat().st_size // 1024
            print(f"  {f.name}  ({size_kb} KB)")


if __name__ == "__main__":
    run_t1_4(verbose=True)
