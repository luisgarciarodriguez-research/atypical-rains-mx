"""
Carga y limpieza inicial del dataset pluviométrico — Tarea T1.1.

Implementa las seis subtareas del paso T1.1:
  T1.1.1  Lectura del CSV con separador de tabulador y sustitución del código
          centinela −99.0 por NaN.
  T1.1.2  Validación y coerción de tipos (Long, Lat, columnas de lluvia m_yyyy).
  T1.1.3  Verificación de coordenadas contra el bounding-box de México y
          generación del mapa de cobertura de estaciones.
  T1.1.4  Comprobación de coherencia entre el campo No.records y el conteo
          real de observaciones válidas por estación.
  T1.1.5  Construcción de columnas auxiliares: pct_complete y years_active.
  T1.1.6  Persistencia del DataFrame limpio en data/processed/lluvia_clean.parquet.

Punto de entrada: ``load_and_clean(verbose=True)``.

Proyecto : Detección de Precipitación Atípica en México (2013–2026)
Unidad   : Universidad Nacional Autónoma de México (UNAM)
           Instituto de Investigaciones en Matemáticas Aplicadas y en Sistemas (IIMAS)
           Posgrado en Ciencia e Ingeniería de la Computación (PCIC)
Investigador líder : Dr. José Antonio Neme Castillo
Contribuidor       : Luis García Rodríguez
"""
import re
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from pathlib import Path
from src.config import (
    RAW_FILE, DATA_PROCESSED, FIGURES, MISSING_CODE, SEPARATOR,
    DROP_COLS, MEXICO_BBOX,
)


def parse_rain_columns(columns):
    """Parsea '1_2013' → (mes=1, año=2013). Retorna dict {col_name: (month, year)}."""
    rain_cols = {}
    for col in columns:
        match = re.match(r'^(\d{1,2})_(\d{4})$', col)
        if match:
            rain_cols[col] = (int(match.group(1)), int(match.group(2)))
    return rain_cols


def load_and_clean(verbose=True):
    """
    T1.1 completo: carga, limpieza, validación y guardado de lluvia_clean.parquet.

    Returns
    -------
    df : pd.DataFrame
        DataFrame limpio con NaN en lugar de -99.
    rain_cols : dict
        {col_name: (month, year)} para todas las columnas de precipitación.
    """
    # ── T1.1.1 Cargar CSV, reemplazar centinelas, eliminar columnas ──
    if verbose:
        print("[T1.1.1] Cargando CSV...")
    df = pd.read_csv(RAW_FILE, sep=SEPARATOR, low_memory=False)
    if verbose:
        print(f"         Shape inicial: {df.shape}")

    df.replace(MISSING_CODE, np.nan, inplace=True)

    existing_drops = [c for c in DROP_COLS if c in df.columns]
    df.drop(columns=existing_drops, inplace=True)
    if verbose:
        print(f"         Columnas eliminadas: {existing_drops}")
        print(f"         Shape tras limpieza: {df.shape}")

    rain_col_map = parse_rain_columns(df.columns)
    rain_cols = list(rain_col_map.keys())
    if verbose:
        print(f"         Columnas de precipitación detectadas: {len(rain_cols)}")

    # ── T1.1.2 Validar tipos ──
    if verbose:
        print("[T1.1.2] Validando tipos...")
    df["Long"] = pd.to_numeric(df["Long"], errors="coerce")
    df["Lat"] = pd.to_numeric(df["Lat"], errors="coerce")
    df["No.records"] = pd.to_numeric(df["No.records"], errors="coerce").astype("Int64")
    for col in rain_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Detectar valores con precisión excesiva (>2 decimales)
    precision_issues = {}
    for col in rain_cols:
        vals = df[col].dropna()
        remainder = (vals - vals.round(2)).abs()
        n_prec = (remainder > 1e-9).sum()
        if n_prec > 0:
            precision_issues[col] = n_prec
    if verbose:
        print(f"         Columnas con precisión excesiva (>2 dec): {len(precision_issues)}")
        if precision_issues:
            top5 = sorted(precision_issues.items(), key=lambda x: -x[1])[:5]
            for c, n in top5:
                print(f"           {c}: {n} valores")

    # ── T1.1.3 Validar coordenadas contra MEXICO_BBOX ──
    if verbose:
        print("[T1.1.3] Validando coordenadas...")
    bbox = MEXICO_BBOX
    out_bbox = (
        (df["Long"] < bbox["lon_min"]) | (df["Long"] > bbox["lon_max"]) |
        (df["Lat"] < bbox["lat_min"]) | (df["Lat"] > bbox["lat_max"])
    )
    missing_coords = df["Long"].isna() | df["Lat"].isna()
    if verbose:
        print(f"         Estaciones fuera del bbox: {out_bbox.sum()}")
        print(f"         Estaciones sin coordenadas: {missing_coords.sum()}")
        if out_bbox.sum() > 0:
            print("         Estaciones fuera de México:")
            print(df.loc[out_bbox, ["#Station", "State", "Long", "Lat"]].to_string())

    _plot_station_coverage(df, out_bbox)

    # ── T1.1.4 Verificar coherencia No.records vs conteo real ──
    if verbose:
        print("[T1.1.4] Verificando coherencia No.records...")
    actual_counts = df[rain_cols].notna().sum(axis=1).astype("Int64")
    df["actual_records"] = actual_counts
    discrepancies = df[df["No.records"] != df["actual_records"]][
        ["#Station", "State", "No.records", "actual_records"]
    ]
    if verbose:
        print(f"         Estaciones con discrepancia: {len(discrepancies)}")
        if len(discrepancies) > 0:
            print(discrepancies.head(10).to_string())

    # ── T1.1.5 Construir columnas auxiliares ──
    if verbose:
        print("[T1.1.5] Construyendo columnas auxiliares...")
    total_months = len(rain_cols)
    df["pct_complete"] = df[rain_cols].notna().sum(axis=1) / total_months

    # years_active: años con al menos 6 meses de datos
    years = sorted(set(y for _, (m, y) in rain_col_map.items()))
    years_active_counts = []
    for idx in df.index:
        count = 0
        for yr in years:
            yr_cols = [c for c, (m, y) in rain_col_map.items() if y == yr]
            if df.loc[idx, yr_cols].notna().sum() >= 6:
                count += 1
        years_active_counts.append(count)
    df["years_active"] = years_active_counts

    if verbose:
        print(f"         pct_complete — media: {df['pct_complete'].mean():.3f}, "
              f"mediana: {df['pct_complete'].median():.3f}")
        print(f"         years_active — media: {df['years_active'].mean():.1f}, "
              f"max: {df['years_active'].max()}")

    # ── T1.1.6 Guardar parquet ──
    if verbose:
        print("[T1.1.6] Guardando lluvia_clean.parquet...")
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out_path = DATA_PROCESSED / "lluvia_clean.parquet"
    df.to_parquet(out_path, index=True)
    if verbose:
        print(f"         Guardado en: {out_path}")
        print(f"         Shape final: {df.shape}")

    # Criterio de aceptación
    rain_values = df[rain_cols].values.flatten()
    rain_valid = rain_values[~np.isnan(rain_values)]
    assert rain_valid.min() >= 0, f"Valores negativos residuales: {rain_valid.min()}"
    if verbose:
        print(f"\n[OK] Criterio de aceptación: min={rain_valid.min():.2f}, "
              f"max={rain_valid.max():.2f}, n_valid={len(rain_valid):,}")

    return df, rain_col_map


def _plot_station_coverage(df, out_bbox_mask):
    """Scatter plot lon/lat de estaciones con contorno de México."""
    mexico = None
    # Intentar cargar shapefile de Natural Earth (formato geodatasets o URL)
    try:
        import geodatasets
        world = gpd.read_file(geodatasets.get_path("naturalearth.land"))
        # Solo contorno tierra — usamos directamente
        mexico = world
    except Exception:
        pass

    if mexico is None:
        try:
            world = gpd.read_file(
                "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
            )
            # Buscar columna con nombre del país
            name_col = next(
                (c for c in world.columns if c.lower() in ("name", "name_long", "sovereignt")),
                None,
            )
            mexico = world[world[name_col] == "Mexico"] if name_col else world
        except Exception:
            pass

    fig, ax = plt.subplots(figsize=(12, 8))

    fig, ax = plt.subplots(figsize=(12, 8))
    if mexico is not None and not mexico.empty:
        mexico.plot(ax=ax, color="lightgray", edgecolor="black", linewidth=0.8)
    else:
        print("         [aviso] Sin shapefile de México; el mapa no tendrá contorno.")

    inside = ~out_bbox_mask & df["Long"].notna()
    outside = out_bbox_mask & df["Long"].notna()

    ax.scatter(df.loc[inside, "Long"], df.loc[inside, "Lat"],
               c="steelblue", s=8, alpha=0.6, label="Dentro del bbox")
    if outside.sum() > 0:
        ax.scatter(df.loc[outside, "Long"], df.loc[outside, "Lat"],
                   c="red", s=20, alpha=0.9, label="Fuera del bbox")

    ax.set_xlim(MEXICO_BBOX["lon_min"] - 2, MEXICO_BBOX["lon_max"] + 2)
    ax.set_ylim(MEXICO_BBOX["lat_min"] - 2, MEXICO_BBOX["lat_max"] + 2)
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.set_title("Cobertura de estaciones pluviométricas (T1.1.3)")
    ax.legend(loc="lower left")

    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / "T1.1.3_station_coverage.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"         Mapa guardado en: {out}")


if __name__ == "__main__":
    df, rain_col_map = load_and_clean(verbose=True)
