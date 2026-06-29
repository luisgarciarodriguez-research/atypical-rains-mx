"""
Preparación del subconjunto analítico y tratamiento de ceros para CoDA — Tareas T3.1 y T3.2.

T3.1 — Selección del subconjunto analítico
  Aplica cuatro filtros en cascada sobre lluvia_clean.parquet para retener
  únicamente las estaciones aptas para el análisis composicional:
    T3.1.2  Excluir estaciones con cero observaciones válidas en toda la serie.
    T3.1.3  Excluir estaciones con coordenadas fuera del bounding-box de México.
    T3.1.4  Recodificar artefactos instrumentales confirmados (catálogo T2.5)
            sustituyendo los valores afectados por NaN.
    T3.1.1  Filtro temporal: ≥10 meses con dato por año y ≥3 años calificantes.
  Resultado: lluvia_coda.parquet (1,302 de 1,959 estaciones, 66.5%).

T3.2 — Tratamiento de ceros en composiciones pluviométricas
  La transformación logarítmica que requiere CoDA es indefinida en cero. Se
  calcula primero la composición media mensual de 12 partes (Σ=1) por estación
  y luego se elimina la restricción de cero mediante:
    T3.2.3  Reemplazo multiplicativo (Martín-Fernández et al., 2003):
            δ = 0.65 × mín(partes no-cero de la fila).
    T3.2.4  Alternativa Bayesiana-Laplace (sin rpy2) para análisis de sensibilidad.
    T3.2.5  Comparación cuantitativa de ambos métodos en las celdas afectadas.
  Resultado: compositions_no_zeros.parquet.

Punto de entrada T3.1: ``run_t3_1()``.
Punto de entrada T3.2: ``run_t3_2()``.

Proyecto : Detección de Precipitación Atípica en México (2013–2026)
Unidad   : Universidad Nacional Autónoma de México (UNAM)
           Instituto de Investigaciones en Matemáticas Aplicadas y en Sistemas (IIMAS)
           Posgrado en Ciencia e Ingeniería de la Computación (PCIC)
Investigador líder : Dr. José Antonio Neme Castillo
Contribuidor       : Luis García Rodríguez
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from src.config import DATA_PROCESSED, DATA_CATALOGS, FIGURES, MEXICO_BBOX
from src.loading import parse_rain_columns


def _sorted_rain_cols(rain_col_map: dict) -> list[str]:
    return sorted(rain_col_map, key=lambda c: (rain_col_map[c][1], rain_col_map[c][0]))


# ── T3.1.1 — Filtro de suficiencia temporal ───────────────────────────────────

def select_coda_subset(
    df: pd.DataFrame,
    rain_col_map: dict,
    min_months_per_year: int = 10,
    min_years: int = 3,
) -> pd.DataFrame:
    """
    T3.1.1 — Filtro de inclusión para análisis composicional.

    Requiere al menos min_months_per_year con dato en al menos min_years años.
    Incluye todos los años (2013–2026) pero 2026 rara vez supera el umbral de
    10 meses al tener solo 5, por lo que en la práctica se basa en 2013–2025.
    """
    years = sorted(set(y for _, (m, y) in rain_col_map.items()))
    station_years: dict = {}

    for idx in df.index:
        qualifying_years = 0
        for year in years:
            year_cols = [c for c, (m, y) in rain_col_map.items() if y == year]
            n_valid = df.loc[idx, year_cols].notna().sum()
            if n_valid >= min_months_per_year:
                qualifying_years += 1
        station_years[idx] = qualifying_years

    mask = pd.Series(station_years) >= min_years
    return df[mask].copy()


# ── T3.1.2 — Excluir estaciones sin datos ─────────────────────────────────────

def exclude_zero_data(df: pd.DataFrame, rain_cols: list[str]) -> pd.DataFrame:
    """
    T3.1.2 — Elimina estaciones con cero valores válidos en toda la serie.

    Equivale a `actual_records == 0` si la columna existe, o bien
    a que todas las columnas de lluvia sean NaN.
    """
    if "actual_records" in df.columns:
        mask = df["actual_records"] > 0
    else:
        mask = df[rain_cols].notna().any(axis=1)
    return df[mask].copy()


# ── T3.1.3 — Excluir coordenadas erróneas ─────────────────────────────────────

def exclude_bad_coords(df: pd.DataFrame) -> pd.DataFrame:
    """
    T3.1.3 — Elimina estaciones con coordenadas fuera del bounding box de México
    o con coordenadas nulas.
    """
    lon_ok = (
        df["Long"].notna()
        & (df["Long"] >= MEXICO_BBOX["lon_min"])
        & (df["Long"] <= MEXICO_BBOX["lon_max"])
    )
    lat_ok = (
        df["Lat"].notna()
        & (df["Lat"] >= MEXICO_BBOX["lat_min"])
        & (df["Lat"] <= MEXICO_BBOX["lat_max"])
    )
    return df[lon_ok & lat_ok].copy()


# ── T3.1.4 — Recodificar artefactos instrumentales ───────────────────────────

def recode_artifacts(
    df: pd.DataFrame,
    catalog_path: Path,
) -> tuple[pd.DataFrame, int]:
    """
    T3.1.4 — Sustituye artefactos instrumentales confirmados por NaN.

    Lee el catálogo de anomalías y aplica `action == 'recodificar_nan'`
    a las celdas correspondientes.

    Retorna (df_recoded, n_recoded).
    """
    catalog = pd.read_csv(catalog_path)
    artifacts = catalog[catalog["action"] == "recodificar_nan"]

    df = df.copy()

    # Índice inverso: #Station → fila(s) del df
    station_to_idx: dict[str, list] = {}
    for idx, st in df["#Station"].items():
        station_to_idx.setdefault(st, []).append(idx)

    n_recoded = 0
    for _, row in artifacts.iterrows():
        st  = row["station"]
        col = row["month_col"]
        if col not in df.columns:
            continue
        indices = station_to_idx.get(st, [])
        for idx in indices:
            if pd.notna(df.at[idx, col]):
                df.at[idx, col] = np.nan
                n_recoded += 1

    return df, n_recoded


# ── Figura resumen T3.1 ───────────────────────────────────────────────────────

def _plot_subset_summary(
    df_orig: pd.DataFrame,
    df_coda: pd.DataFrame,
    rain_col_map: dict,
    rain_cols: list[str],
    filter_log: list[tuple[str, int]],
) -> Path:
    """Cuatro paneles: embudo de filtrado, completitud, mapa, distribución."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # ── Panel 1: embudo de filtros ──
    labels = [f[0] for f in filter_log]
    sizes  = [f[1] for f in filter_log]
    colors = plt.cm.Blues(np.linspace(0.35, 0.85, len(sizes)))
    bars = axes[0, 0].barh(labels[::-1], sizes[::-1], color=colors[::-1],
                            edgecolor="white")
    for bar, n in zip(bars, sizes[::-1]):
        axes[0, 0].text(bar.get_width() + max(sizes) * 0.01, bar.get_y() + bar.get_height() / 2,
                        f"{n:,}", va="center", fontsize=9)
    axes[0, 0].set_title("CoDA subset selection funnel", fontsize=10)
    axes[0, 0].set_xlabel("No. of stations")
    axes[0, 0].set_xlim(0, max(sizes) * 1.15)

    # ── Panel 2: distribución de % completitud en el subconjunto ──
    pct = df_coda[rain_cols].notna().mean(axis=1) * 100
    axes[0, 1].hist(pct, bins=40, color="#1976D2", alpha=0.8, edgecolor="white")
    axes[0, 1].axvline(pct.median(), color="red", linestyle="--",
                       label=f"Mediana={pct.median():.1f}%")
    axes[0, 1].set_title(
        f"Temporal series completeness\n(CoDA subset: {len(df_coda):,} stations)",
        fontsize=10,
    )
    axes[0, 1].set_xlabel("% months with valid data")
    axes[0, 1].set_ylabel("No. of stations")
    axes[0, 1].legend(fontsize=9)

    # ── Panel 3: mapa — incluidas vs excluidas ──
    included = df_orig.index.isin(df_coda.index)
    axes[1, 0].scatter(
        df_orig.loc[~included, "Long"], df_orig.loc[~included, "Lat"],
        color="#BDBDBD", s=5, alpha=0.5, label="Excluded",
    )
    axes[1, 0].scatter(
        df_coda["Long"], df_coda["Lat"],
        color="#1565C0", s=8, alpha=0.7, label=f"Included ({len(df_coda):,})",
    )
    axes[1, 0].set_title("Spatial distribution of CoDA subset", fontsize=10)
    axes[1, 0].set_xlabel("Longitude")
    axes[1, 0].set_ylabel("Latitude")
    axes[1, 0].legend(fontsize=8, markerscale=2)

    # ── Panel 4: años calificantes por estación ──
    years = sorted({y for _, (_, y) in rain_col_map.items() if y < 2026})
    qy_vals = []
    for idx in df_coda.index:
        qy = sum(
            1 for yr in years
            if df_coda.loc[idx, [c for c, (_, y) in rain_col_map.items() if y == yr]].notna().sum() >= 10
        )
        qy_vals.append(qy)
    axes[1, 1].hist(qy_vals, bins=range(0, len(years) + 2), color="#1565C0",
                    alpha=0.8, edgecolor="white", align="left")
    axes[1, 1].axvline(3, color="red", linestyle="--", linewidth=1.2,
                       label="min. threshold (3 years)")
    axes[1, 1].set_title("Qualifying years (≥10 months/year) per station", fontsize=10)
    axes[1, 1].set_xlabel("No. of years with ≥10 valid months")
    axes[1, 1].set_ylabel("No. of stations")
    axes[1, 1].legend(fontsize=9)

    fig.suptitle("T3.1 — Analytical Subset Selection for CoDA", fontsize=12)
    fig.tight_layout()

    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / "coda_subset_selection.png"
    fig.savefig(out, dpi=900, bbox_inches="tight")
    plt.close(fig)
    return out


# ── Punto de entrada T3.1 ─────────────────────────────────────────────────────

def run_t3_1(
    min_months_per_year: int = 10,
    min_years: int = 3,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Ejecuta T3.1 completo.

    Genera:
      - data/processed/lluvia_coda.parquet   (subconjunto filtrado y limpio)
      - outputs/figures/coda_subset_selection.png

    Retorna el DataFrame del subconjunto CoDA.
    """
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("=" * 60)
        print("T3.1 — Selección del Subconjunto Analítico para CoDA")
        print("=" * 60)

    df_orig = pd.read_parquet(DATA_PROCESSED / "lluvia_clean.parquet")
    rain_col_map = parse_rain_columns(df_orig.columns)
    rain_cols = _sorted_rain_cols(rain_col_map)

    if verbose:
        print(f"\nPunto de partida: {len(df_orig):,} estaciones × {len(rain_cols)} meses")

    filter_log: list[tuple[str, int]] = [
        ("Inicial (lluvia_clean)", len(df_orig)),
    ]

    # ── T3.1.2 — Excluir 0-datos ─────────────────────────────────────────────
    df = exclude_zero_data(df_orig, rain_cols)
    if verbose:
        n_removed = len(df_orig) - len(df)
        print(f"\n[T3.1.2] Excluir estaciones sin datos: −{n_removed}  → {len(df):,}")
    filter_log.append(("T3.1.2 Excluir 0-datos", len(df)))

    # ── T3.1.3 — Excluir coords erróneas ─────────────────────────────────────
    prev = len(df)
    df = exclude_bad_coords(df)
    if verbose:
        n_removed = prev - len(df)
        print(f"[T3.1.3] Excluir coords erróneas: −{n_removed}  → {len(df):,}")
    filter_log.append(("T3.1.3 Excluir coords erróneas", len(df)))

    # ── T3.1.4 — Recodificar artefactos ──────────────────────────────────────
    catalog_path = DATA_CATALOGS / "anomalias_catalogo.csv"
    df, n_recoded = recode_artifacts(df, catalog_path)
    if verbose:
        print(f"[T3.1.4] Recodificar artefactos → NaN: {n_recoded:,} celdas")
    filter_log.append(("T3.1.4 Post-recodificación artefactos", len(df)))

    # ── T3.1.1 — Filtro temporal ─────────────────────────────────────────────
    if verbose:
        print(f"[T3.1.1] Filtro: ≥{min_months_per_year} meses × ≥{min_years} años...")
    prev = len(df)
    df_coda = select_coda_subset(df, rain_col_map,
                                  min_months_per_year=min_months_per_year,
                                  min_years=min_years)
    n_removed = prev - len(df_coda)
    if verbose:
        print(f"         −{n_removed:,} estaciones  → {len(df_coda):,} en subconjunto")
    filter_log.append((f"T3.1.1 ≥{min_months_per_year}m/≥{min_years}a", len(df_coda)))

    # ── T3.1.5 — Reporte ─────────────────────────────────────────────────────
    if verbose:
        pct_complete = df_coda[rain_cols].notna().mean(axis=1) * 100
        n_states = df_coda["State"].nunique()
        print(f"\n── Subconjunto CoDA resultante ──")
        print(f"  Estaciones:      {len(df_coda):,}")
        print(f"  Estados:         {n_states}")
        print(f"  Completitud med: {pct_complete.median():.1f}%  "
              f"[{pct_complete.min():.1f}% – {pct_complete.max():.1f}%]")

        years_full = sorted({y for _, (_, y) in rain_col_map.items() if y < 2026})
        qy_all = []
        for idx in df_coda.index:
            qy = sum(
                1 for yr in years_full
                if df_coda.loc[idx,
                    [c for c, (_, y) in rain_col_map.items() if y == yr]
                   ].notna().sum() >= min_months_per_year
            )
            qy_all.append(qy)
        qy_s = pd.Series(qy_all)
        print(f"  Años calificantes: med={qy_s.median():.0f}  "
              f"[{qy_s.min()} – {qy_s.max()}]")

        # Distribución por estado (top-10 y bottom-5)
        per_state = df_coda["State"].value_counts()
        print(f"\n  Top-10 estados por estaciones incluidas:")
        for st, n in per_state.head(10).items():
            print(f"    {st}: {n}")
        print(f"  Bottom-5 estados:")
        for st, n in per_state.tail(5).items():
            print(f"    {st}: {n}")

    # ── Guardar ───────────────────────────────────────────────────────────────
    out_path = DATA_PROCESSED / "lluvia_coda.parquet"
    df_coda.to_parquet(out_path)
    if verbose:
        print(f"\n[Guardado] {out_path}")
        print(f"           {df_coda.shape[0]:,} filas × {df_coda.shape[1]} columnas")

    # ── Figura ────────────────────────────────────────────────────────────────
    fig_path = _plot_subset_summary(df_orig, df_coda, rain_col_map,
                                     rain_cols, filter_log)
    if verbose:
        print(f"[Figura]  {fig_path.name}")

    print("\n[OK] T3.1 completado.")
    return df_coda


# ═══════════════════════════════════════════════════════════════════════════════
# T3.2 — TRATAMIENTO DE CEROS
# ═══════════════════════════════════════════════════════════════════════════════

MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]


# ── T3.2.1 — Composiciones brutas ─────────────────────────────────────────────

def compute_compositions(
    df: pd.DataFrame,
    rain_col_map: dict,
) -> pd.DataFrame:
    """
    T3.2.1 — Composición media mensual (12 partes) por estación.

    Para cada mes calendárico, promedia los valores de todos los años
    disponibles (media ignorando NaN). Las medias NaN (mes sin ningún
    dato en toda la serie) se tratan como 0 antes de normalizar, bajo
    la interpretación de cero estructural (contribución nula al total).

    Retorna DataFrame (estaciones × 12), columnas 1..12, Σfila=1.
    """
    monthly_means = pd.DataFrame(index=df.index, columns=range(1, 13), dtype=float)
    for month in range(1, 13):
        month_cols = [c for c, (m, y) in rain_col_map.items() if m == month]
        monthly_means[month] = df[month_cols].mean(axis=1)

    # NaN estructural → 0 (mes sin ningún dato tratado como cero rainfall)
    monthly_means = monthly_means.fillna(0.0)

    row_sums = monthly_means.sum(axis=1)
    compositions = monthly_means.div(row_sums, axis=0)
    return compositions


# ── T3.2.2 — Diagnóstico de ceros ─────────────────────────────────────────────

def diagnose_zeros(compositions: pd.DataFrame) -> dict:
    """
    T3.2.2 — Cuenta y ubica ceros en la matriz de composiciones.

    Un cero en la composición indica que la media mensual fue 0 mm
    (ya sea medición de 0 o mes sin datos → cero estructural).

    Retorna diccionario con estadísticos de diagnóstico.
    """
    zero_mask = compositions == 0.0
    n_zeros_per_station = zero_mask.sum(axis=1)
    n_zeros_per_month   = zero_mask.sum(axis=0)

    return {
        "total_zeros": int(zero_mask.values.sum()),
        "stations_with_zeros": int((n_zeros_per_station > 0).sum()),
        "zeros_per_month": n_zeros_per_month.to_dict(),
        "zeros_per_station_dist": n_zeros_per_station.value_counts().sort_index().to_dict(),
        "zero_mask": zero_mask,
        "pct_cells_zero": float(zero_mask.values.sum()) / zero_mask.size * 100,
    }


# ── T3.2.3 — Reemplazo multiplicativo ─────────────────────────────────────────

def multiplicative_replacement(
    comp: pd.DataFrame,
    delta_factor: float = 0.65,
) -> pd.DataFrame:
    """
    T3.2.3 — Reemplazo multiplicativo de ceros (Martín-Fernández et al., 2003).

    Para cada estación con ≥1 cero:
      delta = delta_factor × min(partes no-cero de esa fila)
      zeros  → delta
      no-zeros → no-zeros × (1 − n_zeros × delta) / sum(no-zeros)
    Mantiene Σ=1 por construcción.
    """
    result = comp.copy()
    for idx in result.index:
        row = result.loc[idx].values.astype(float)
        zeros = row == 0
        if not zeros.any():
            continue
        non_zero_min = row[~zeros].min()
        delta = delta_factor * non_zero_min
        n_zeros = int(zeros.sum())
        row[zeros] = delta
        row[~zeros] = row[~zeros] * (1 - n_zeros * delta) / row[~zeros].sum()
        result.loc[idx] = row
    return result


# ── T3.2.4 — Alternativa Bayesiana (sin R/rpy2) ───────────────────────────────

def bayesian_laplace_replacement(
    comp: pd.DataFrame,
    alpha: float = 0.65,
) -> pd.DataFrame:
    """
    T3.2.4 alternativa — Reemplazo Bayesiano-Laplace (Palarea-Albaladejo &
    Martín-Fernández, 2008, versión simplificada).

    Sustituye ceros por alpha × (1/D), donde D=12 es el número de partes,
    luego re-normaliza para mantener Σ=1.  A diferencia del reemplazo
    multiplicativo, usa un delta fijo relativo a la composición uniforme
    (1/12), independiente de los valores no-cero de la fila.

    Se usa exclusivamente para el análisis de sensibilidad (T3.2.5).
    Nota: lrEM (zCompositions::lrEM) es el método preferido en R pero
    rpy2 no está disponible en este entorno.
    """
    D = comp.shape[1]
    result = comp.copy()
    for idx in result.index:
        row = result.loc[idx].values.astype(float)
        zeros = row == 0
        if not zeros.any():
            continue
        delta = alpha / D
        n_zeros = int(zeros.sum())
        row[zeros] = delta
        row[~zeros] = row[~zeros] * (1 - n_zeros * delta) / row[~zeros].sum()
        result.loc[idx] = row
    return result


# ── T3.2.5 — Análisis de sensibilidad ────────────────────────────────────────

def sensitivity_analysis(
    comp_mult: pd.DataFrame,
    comp_bayes: pd.DataFrame,
    zero_mask: pd.DataFrame,
) -> pd.DataFrame:
    """
    T3.2.5 — Compara las dos estrategias de reemplazo en las celdas afectadas.

    Métricas por estación con ≥1 cero:
      - max_diff  : máxima diferencia absoluta entre los dos métodos (en cualquier parte)
      - perturbation : distancia euclidiana entre los vectores resultantes (escala composición)

    Retorna DataFrame con métricas por estación (solo las con ceros).
    """
    stations_with_zeros = zero_mask.index[zero_mask.any(axis=1)]
    records = []
    for idx in stations_with_zeros:
        a = comp_mult.loc[idx].values.astype(float)
        b = comp_bayes.loc[idx].values.astype(float)
        records.append({
            "idx": idx,
            "n_zeros": int(zero_mask.loc[idx].sum()),
            "max_diff": float(np.abs(a - b).max()),
            "euclidean_dist": float(np.sqrt(((a - b)**2).sum())),
        })
    return pd.DataFrame(records).set_index("idx")


# ── Figura T3.2 ───────────────────────────────────────────────────────────────

def _plot_zero_treatment(
    compositions_raw: pd.DataFrame,
    comp_mult: pd.DataFrame,
    comp_bayes: pd.DataFrame,
    diag: dict,
    sensitivity: pd.DataFrame,
) -> Path:
    """Cuatro paneles: mapa de ceros, distribuciones por mes, sensibilidad, Σ=1 check."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # ── Panel 1: heatmap de ceros por mes ──
    zero_by_month = pd.Series(diag["zeros_per_month"])
    bar_colors = [
        "#D32F2F" if m in [5,6,7,8,9,10] else "#1565C0"
        for m in range(1, 13)
    ]
    axes[0, 0].bar(MONTH_NAMES, [zero_by_month.get(m, 0) for m in range(1, 13)],
                   color=bar_colors, edgecolor="white")
    axes[0, 0].set_title(
        f"Zeros in raw compositions by month\n"
        f"Total: {diag['total_zeros']} cells in {diag['stations_with_zeros']} stations "
        f"({diag['pct_cells_zero']:.2f}%)",
        fontsize=10,
    )
    axes[0, 0].set_ylabel("No. of stations with zero")
    axes[0, 0].set_xlabel("Month (red=wet, blue=dry)")

    # ── Panel 2: distribución de composiciones antes / después (mes con más ceros = abril) ──
    ref_month = max(diag["zeros_per_month"], key=lambda m: diag["zeros_per_month"][m])
    raw_vals  = compositions_raw[ref_month].values
    mult_vals = comp_mult[ref_month].values
    valid_raw  = raw_vals[~np.isnan(raw_vals) & (raw_vals > 0)]
    valid_mult = mult_vals[mult_vals > 0]
    axes[0, 1].hist(valid_raw,  bins=40, alpha=0.5, color="#1976D2", label="Raw (without zeros)")
    axes[0, 1].hist(valid_mult, bins=40, alpha=0.5, color="#E53935", label="Post-multiplicative replacement")
    axes[0, 1].set_title(f"Composition distribution — {MONTH_NAMES[ref_month-1]} "
                          f"(month with most zeros)", fontsize=10)
    axes[0, 1].set_xlabel("Proportion")
    axes[0, 1].set_ylabel("No. of stations")
    axes[0, 1].legend(fontsize=8)

    # ── Panel 3: análisis de sensibilidad (mult vs Bayesiano) ──
    if not sensitivity.empty:
        axes[1, 0].scatter(sensitivity["n_zeros"], sensitivity["max_diff"],
                           c=sensitivity["euclidean_dist"], cmap="YlOrRd",
                           s=30, alpha=0.7, edgecolors="none")
        axes[1, 0].set_xlabel("No. of zeros in composition")
        axes[1, 0].set_ylabel("Max. absolute difference |mult − Bayesian|")
        axes[1, 0].set_title(
            "Sensitivity: Multiplicative vs Bayesian-Laplace replacement\n"
            f"(stations with zeros: {len(sensitivity)})",
            fontsize=10,
        )
        sm = plt.cm.ScalarMappable(cmap="YlOrRd")
        sm.set_array(sensitivity["euclidean_dist"].values)
        plt.colorbar(sm, ax=axes[1, 0], label="Euclidean dist.")

    # ── Panel 4: mediana de composición por mes (antes y después) ──
    medians_raw  = [compositions_raw[m].median() for m in range(1, 13)]
    medians_mult = [comp_mult[m].median()        for m in range(1, 13)]
    x = np.arange(12)
    w = 0.35
    axes[1, 1].bar(x - w/2, medians_raw,  width=w, label="Raw",
                   color="#1976D2", alpha=0.7, edgecolor="white")
    axes[1, 1].bar(x + w/2, medians_mult, width=w, label="Post-multiplicative replacement",
                   color="#E53935", alpha=0.7, edgecolor="white")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(MONTH_NAMES, fontsize=8)
    axes[1, 1].axhline(1/12, color="grey", linestyle=":", linewidth=1,
                       label="Uniform composition (1/12)")
    axes[1, 1].set_title(
        "Median composition by month\n(Raw vs Multiplicative replacement)",
        fontsize=10,
    )
    axes[1, 1].set_ylabel("Median proportion")
    axes[1, 1].legend(fontsize=8)

    fig.suptitle("T3.2 — Zero Treatment in Pluviometric Compositions", fontsize=12)
    fig.tight_layout()

    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / "coda_zero_treatment.png"
    fig.savefig(out, dpi=900, bbox_inches="tight")
    plt.close(fig)
    return out


# ── Punto de entrada T3.2 ─────────────────────────────────────────────────────

def run_t3_2(verbose: bool = True) -> pd.DataFrame:
    """
    Ejecuta T3.2 completo.

    Genera:
      - data/processed/compositions_raw.csv          (composiciones antes de tratar ceros)
      - data/processed/compositions_no_zeros.parquet (composiciones post-tratamiento)
      - outputs/figures/coda_zero_treatment.png

    Retorna el DataFrame de composiciones sin ceros (reemplazo multiplicativo).
    """
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("=" * 60)
        print("T3.2 — Tratamiento de Ceros")
        print("=" * 60)

    df = pd.read_parquet(DATA_PROCESSED / "lluvia_coda.parquet")
    rain_col_map = parse_rain_columns(df.columns)

    if verbose:
        print(f"\nSubconjunto CoDA: {len(df):,} estaciones")

    # ── T3.2.1 — Composiciones brutas ────────────────────────────────────────
    if verbose:
        print("\n[T3.2.1] Calculando composiciones mensuales (media × años → Σ=1)...")
    compositions_raw = compute_compositions(df, rain_col_map)

    # Guardar composiciones brutas (también útil para lrEM en R)
    raw_csv = DATA_PROCESSED / "compositions_raw.csv"
    compositions_raw.to_csv(raw_csv)
    if verbose:
        print(f"  Guardado: {raw_csv.name}")
        row_sums = compositions_raw.sum(axis=1)
        print(f"  Verificación Σ=1: min={row_sums.min():.6f}  max={row_sums.max():.6f}")

    # ── T3.2.2 — Diagnóstico de ceros ────────────────────────────────────────
    if verbose:
        print("\n[T3.2.2] Diagnóstico de ceros en composiciones...")
    diag = diagnose_zeros(compositions_raw)

    if verbose:
        print(f"  Total celdas cero: {diag['total_zeros']:,} "
              f"({diag['pct_cells_zero']:.2f}% de 1,302×12={1302*12:,})")
        print(f"  Estaciones con ≥1 cero: {diag['stations_with_zeros']} "
              f"({diag['stations_with_zeros']/len(compositions_raw)*100:.1f}%)")
        print(f"  Ceros por mes:")
        for m, n in diag["zeros_per_month"].items():
            if n > 0:
                print(f"    {MONTH_NAMES[m-1]:>3}: {n:>3} estaciones")
        print(f"  Distribución de ceros por estación:")
        for n_z, count in sorted(diag["zeros_per_station_dist"].items()):
            if n_z > 0:
                print(f"    {n_z} cero(s): {count} estaciones")

    # ── T3.2.3 — Reemplazo multiplicativo ────────────────────────────────────
    if verbose:
        print("\n[T3.2.3] Reemplazo multiplicativo (Martín-Fernández et al., 2003)...")
        print("  delta = 0.65 × min(partes no-cero de la fila)")
    comp_mult = multiplicative_replacement(compositions_raw, delta_factor=0.65)

    n_remaining_zeros = (comp_mult == 0).values.sum()
    if verbose:
        row_sums_m = comp_mult.sum(axis=1)
        print(f"  Ceros residuales: {n_remaining_zeros}")
        print(f"  Verificación Σ=1: min={row_sums_m.min():.8f}  "
              f"max={row_sums_m.max():.8f}  std={row_sums_m.std():.2e}")

    # ── T3.2.4 — Alternativa (sin rpy2) ──────────────────────────────────────
    if verbose:
        print("\n[T3.2.4] Alternativa Bayesiana-Laplace (rpy2/lrEM no disponible)...")
        print("  Usando delta = 0.65/12 por parte (independiente de los valores de la fila)")
    comp_bayes = bayesian_laplace_replacement(compositions_raw, alpha=0.65)

    # ── T3.2.5 — Análisis de sensibilidad ────────────────────────────────────
    if verbose:
        print("\n[T3.2.5] Análisis de sensibilidad (mult. vs Bayesiano)...")
    sens = sensitivity_analysis(comp_mult, comp_bayes, diag["zero_mask"])

    if verbose and not sens.empty:
        print(f"  Estaciones con ceros comparadas: {len(sens)}")
        print(f"  Diferencia máxima absoluta: "
              f"med={sens['max_diff'].median():.6f}  "
              f"max={sens['max_diff'].max():.6f}")
        print(f"  Distancia euclidiana: "
              f"med={sens['euclidean_dist'].median():.6f}  "
              f"max={sens['euclidean_dist'].max():.6f}")
        # Proporción de celdas afectadas donde la diferencia < 0.005
        pct_small = (sens["max_diff"] < 0.005).mean() * 100
        print(f"  Estaciones con diff_max < 0.005: {pct_small:.1f}%  "
              f"(baja sensibilidad al método de reemplazo)")

    # ── Guardar resultado principal ───────────────────────────────────────────
    out_path = DATA_PROCESSED / "compositions_no_zeros.parquet"
    comp_mult.to_parquet(out_path)
    if verbose:
        print(f"\n[Guardado] {out_path}")
        print(f"           {comp_mult.shape[0]:,} estaciones × {comp_mult.shape[1]} meses")

    # ── Figura ────────────────────────────────────────────────────────────────
    fig_path = _plot_zero_treatment(
        compositions_raw, comp_mult, comp_bayes, diag, sens
    )
    if verbose:
        print(f"[Figura]  {fig_path.name}")

    # ── Resumen final ─────────────────────────────────────────────────────────
    if verbose:
        print("\n── Resumen composiciones finales (reemplazo multiplicativo) ──")
        print(f"  Shape: {comp_mult.shape}")
        print(f"  Ceros: {(comp_mult==0).values.sum()} (objetivo: 0)")
        print(f"  Rango: [{comp_mult.values.min():.6f}, {comp_mult.values.max():.6f}]")
        print(f"  Mediana por mes:")
        for m in range(1, 13):
            med = comp_mult[m].median()
            print(f"    {MONTH_NAMES[m-1]:>3}: {med:.4f}  ({med*100:.2f}%)")

    print("\n[OK] T3.2 completado.")
    return comp_mult


if __name__ == "__main__":
    run_t3_2(verbose=True)
