"""
Detección de anomalías pluviométricas por capas — Tareas T2.1 a T2.4.

Implementa cuatro capas de detección con lógica de consenso posterior (T2.5):

  T2.1  Artefactos instrumentales:
          - Ceros sospechosos en meses húmedos (mayo–octubre) según cuantil
            espacial y umbral configurable.
          - Rachas de valores constantes ≥ N meses consecutivos.
          - Valores repetidos idénticos en múltiples estaciones del mismo día.

  T2.2  Valores atípicos univariados:
          - Z-scores por estación y por mes-calendario (z > 3.0).
          - Percentiles extremos dentro de la distribución mensual regional
            (p < 1 o p > 99).
          - Isolation Forest por columna temporal (contamination = 0.02).

  T2.3  Anomalías espaciales:
          - LOF (Local Outlier Factor, n_neighbors = 20) sobre coordenadas
            (lon, lat, valor) normalizadas en cada período.
          - LISA (Índice de Moran Local) para identificar outliers HH/LL/HL/LH
            mediante distancia inversa ponderada (vecinos k = 8).

  T2.4  Anomalías de perfil multivariado:
          - Distancia de Mahalanobis sobre el vector mensual de 12 partes
            por estación; umbral χ²(0.975, 12).
          - Isolation Forest multivariado sobre perfiles completos.

Punto de entrada: ``run_t2_detection(verbose=True)``.

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

from src.config import (
    DATA_PROCESSED, DATA_CATALOGS, FIGURES,
    WET_MONTHS, MISSING_CODE,
)
from src.loading import parse_rain_columns


# ── Utilidades ────────────────────────────────────────────────────────────────

def _sorted_rain_cols(rain_col_map: dict) -> list[str]:
    """Columnas de lluvia ordenadas cronológicamente (año, mes)."""
    return sorted(rain_col_map, key=lambda c: (rain_col_map[c][1], rain_col_map[c][0]))


# ═══════════════════════════════════════════════════════════════════════════════
# T2.1 — CAPA 1: Artefactos Instrumentales
# ═══════════════════════════════════════════════════════════════════════════════

def flag_suspicious_zeros(
    df: pd.DataFrame,
    rain_cols: list[str],
    rain_col_map: dict,
    wet_threshold_mm: float = 50.0,
) -> pd.DataFrame:
    """
    T2.1.1 — Ceros sospechosos.

    Flag = 1 cuando la precipitación es 0.0 en un mes húmedo (mayo–oct)
    y la mediana regional (mismo estado, misma columna) supera wet_threshold_mm.

    Retorna DataFrame booleano (estaciones × rain_cols).
    """
    flags = pd.DataFrame(False, index=df.index, columns=rain_cols)

    for col, (month, year) in rain_col_map.items():
        if month not in WET_MONTHS:
            continue

        for state in df["State"].unique():
            state_mask = df["State"] == state
            regional_median = df.loc[state_mask, col].median()

            if pd.isna(regional_median) or regional_median <= wet_threshold_mm:
                continue

            zero_mask = state_mask & (df[col] == 0.0)
            flags.loc[zero_mask, col] = True

    return flags


def flag_stuck_sensor(
    df: pd.DataFrame,
    rain_cols: list[str],
    min_consecutive: int = 3,
) -> pd.DataFrame:
    """
    T2.1.2 — Sensor atascado.

    Detecta ≥ min_consecutive meses consecutivos con el mismo valor
    no-nulo y no-cero por estación.
    rain_cols deben estar en orden cronológico.

    Retorna DataFrame booleano (estaciones × rain_cols).
    """
    flags = pd.DataFrame(False, index=df.index, columns=rain_cols)

    for i, idx in enumerate(df.index):
        vals = df.loc[idx, rain_cols].values.astype(float)
        count = 1
        for j in range(1, len(vals)):
            if (
                vals[j] == vals[j - 1]
                and not np.isnan(vals[j])
                and vals[j] != 0.0
            ):
                count += 1
                if count >= min_consecutive:
                    # marcar todo el bloque actual
                    for k in range(j - count + 1, j + 1):
                        flags.iat[i, k] = True
            else:
                count = 1

    return flags


def flag_negative_residuals(
    df: pd.DataFrame, rain_cols: list[str]
) -> pd.DataFrame:
    """
    T2.1.3 — Valores negativos residuales.

    Verifica que no queden valores < 0 tras la limpieza de T1.1.
    Retorna DataFrame booleano; en datasets ya limpios será todo False.
    """
    flags = pd.DataFrame(False, index=df.index, columns=rain_cols)
    for col in rain_cols:
        neg_mask = df[col] < 0
        flags.loc[neg_mask, col] = True
    return flags


def flag_precision_anomalies(
    df: pd.DataFrame,
    rain_cols: list[str],
    max_decimals: int = 2,
) -> pd.DataFrame:
    """
    T2.1.4 — Precisión anómala.

    Valores como 104.1500015 indican errores de punto flotante en la
    cadena de digitalización.  Flag cuando |round(x, max_decimals) - x| > 1e-9.

    Retorna DataFrame booleano (estaciones × rain_cols).
    """
    flags = pd.DataFrame(False, index=df.index, columns=rain_cols)

    for col in rain_cols:
        vals = df[col]
        valid_mask = vals.notna()
        remainder = (vals[valid_mask] - vals[valid_mask].round(max_decimals)).abs()
        anomalous = remainder > 1e-9
        flags.loc[anomalous.index[anomalous], col] = True

    return flags


# ── Combinar y guardar Capa 1 ─────────────────────────────────────────────────

def _summary_table(
    name: str,
    flag_df: pd.DataFrame,
    rain_col_map: dict,
    df: pd.DataFrame,
) -> dict:
    """Estadísticos de resumen de un DataFrame de flags."""
    total = flag_df.size
    n_flagged = int(flag_df.sum().sum())
    n_stations = int((flag_df.sum(axis=1) > 0).sum())
    n_cols     = int((flag_df.sum(axis=0) > 0).sum())
    return {
        "method": name,
        "cells_flagged": n_flagged,
        "pct_of_valid": n_flagged / total * 100,
        "stations_affected": n_stations,
        "months_affected": n_cols,
    }


def _plot_capa1_summary(
    flag_dfs: dict[str, pd.DataFrame],
    rain_cols: list[str],
    rain_col_map: dict,
) -> Path:
    """Figura resumen de flags de Capa 1 por método y distribución temporal."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    COLORS = {
        "zeros": "#2196F3",
        "stuck": "#FF9800",
        "negative": "#9C27B0",
        "precision": "#F44336",
        "combined": "#212121",
    }

    # Panel 1: conteo total de flags por método
    methods = list(flag_dfs.keys())
    counts  = [int(flag_dfs[m].sum().sum()) for m in methods]
    bars = axes[0, 0].bar(methods, counts,
                          color=[COLORS.get(m, "gray") for m in methods],
                          edgecolor="white", width=0.6)
    for bar, cnt in zip(bars, counts):
        axes[0, 0].text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + max(counts) * 0.01,
                        f"{cnt:,}", ha="center", fontsize=9)
    axes[0, 0].set_title("Flagged cells by method (T2.1)", fontsize=10)
    axes[0, 0].set_ylabel("No. of cells")

    # Panel 2: estaciones afectadas por método
    n_st = [(flag_dfs[m].sum(axis=1) > 0).sum() for m in methods]
    axes[0, 1].bar(methods, n_st,
                   color=[COLORS.get(m, "gray") for m in methods],
                   edgecolor="white", width=0.6)
    for i, n in enumerate(n_st):
        axes[0, 1].text(i, n + 5, str(n), ha="center", fontsize=9)
    axes[0, 1].set_title("Stations affected by method", fontsize=10)
    axes[0, 1].set_ylabel("No. of stations")

    # Panel 3: flags temporales del flag combinado (flags/mes)
    combined = flag_dfs.get("combined", flag_dfs[methods[0]])
    monthly_flags = combined.sum(axis=0)

    year_ticks = {i: rain_col_map[c][1]
                  for i, c in enumerate(rain_cols) if rain_col_map[c][0] == 1}
    axes[1, 0].bar(range(len(rain_cols)), monthly_flags.values,
                   color="#212121", alpha=0.75, width=1)
    axes[1, 0].set_xticks(list(year_ticks.keys()))
    axes[1, 0].set_xticklabels([str(y) for y in year_ticks.values()],
                                rotation=45, fontsize=8)
    axes[1, 0].set_title("Temporal distribution — Layer 1 flags (combined)", fontsize=10)
    axes[1, 0].set_ylabel("No. of flagged cells")
    axes[1, 0].set_xlabel("Month")

    # Panel 4: distribución por estación (n flags por estación)
    station_counts = combined.sum(axis=1)
    axes[1, 1].hist(station_counts[station_counts > 0],
                    bins=40, color="#212121", alpha=0.8, edgecolor="white")
    axes[1, 1].set_title("Flag distribution by station\n(only stations with ≥1 flag)", fontsize=10)
    axes[1, 1].set_xlabel("No. of flagged months per station")
    axes[1, 1].set_ylabel("No. of stations")

    fig.suptitle("T2.1 — Layer 1 Summary: Instrumental Artifacts", fontsize=12)
    fig.tight_layout()

    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / "anomaly_capa1_summary.png"
    fig.savefig(out, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return out


# ── Punto de entrada T2.1 ────────────────────────────────────────────────────

def run_t2_1(verbose: bool = True) -> pd.DataFrame:
    """
    Ejecuta T2.1 completo.

    Genera ``data/catalogs/flags_capa1.parquet`` y figura de resumen.
    Retorna el DataFrame de flags combinado (estaciones × rain_cols).
    """
    DATA_CATALOGS.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("=" * 60)
        print("T2.1 — Capa 1: Artefactos Instrumentales")
        print("=" * 60)

    df = pd.read_parquet(DATA_PROCESSED / "lluvia_clean.parquet")
    rain_col_map = parse_rain_columns(df.columns)
    rain_cols = _sorted_rain_cols(rain_col_map)

    if verbose:
        print(f"Datos: {df.shape[0]:,} estaciones × {len(rain_cols)} meses\n")

    # ── T2.1.1 — Ceros sospechosos ──────────────────────────────────────────
    if verbose:
        print("[T2.1.1] Ceros sospechosos en meses húmedos...")
    f_zeros = flag_suspicious_zeros(df, rain_cols, rain_col_map)
    s_zeros = _summary_table("zeros", f_zeros, rain_col_map, df)
    if verbose:
        print(f"         {s_zeros['cells_flagged']:,} celdas  "
              f"({s_zeros['pct_of_valid']:.2f}% del total)  "
              f"en {s_zeros['stations_affected']:,} estaciones")

    # ── T2.1.2 — Sensor atascado ─────────────────────────────────────────────
    if verbose:
        print("[T2.1.2] Sensor atascado (≥3 meses consecutivos mismo valor)...")
    f_stuck = flag_stuck_sensor(df, rain_cols, min_consecutive=3)
    s_stuck = _summary_table("stuck", f_stuck, rain_col_map, df)
    if verbose:
        print(f"         {s_stuck['cells_flagged']:,} celdas  "
              f"({s_stuck['pct_of_valid']:.2f}%)  "
              f"en {s_stuck['stations_affected']:,} estaciones")

    # ── T2.1.3 — Negativos residuales ───────────────────────────────────────
    if verbose:
        print("[T2.1.3] Valores negativos residuales...")
    f_neg = flag_negative_residuals(df, rain_cols)
    s_neg = _summary_table("negative", f_neg, rain_col_map, df)
    if verbose:
        print(f"         {s_neg['cells_flagged']:,} celdas "
              f"({'ninguna — limpieza T1.1 correcta' if s_neg['cells_flagged']==0 else 'ATENCIÓN'})")

    # ── T2.1.4 — Precisión anómala ───────────────────────────────────────────
    if verbose:
        print("[T2.1.4] Precisión anómala (>2 decimales)...")
    f_prec = flag_precision_anomalies(df, rain_cols, max_decimals=2)
    s_prec = _summary_table("precision", f_prec, rain_col_map, df)
    if verbose:
        print(f"         {s_prec['cells_flagged']:,} celdas  "
              f"({s_prec['pct_of_valid']:.2f}%)  "
              f"en {s_prec['stations_affected']:,} estaciones  "
              f"en {s_prec['months_affected']} columnas-mes")

    # ── Combinar (OR) ────────────────────────────────────────────────────────
    if verbose:
        print("\n[Combinando flags Capa 1 (OR)...]")
    flags_capa1 = f_zeros | f_stuck | f_neg | f_prec
    s_combined = _summary_table("combined", flags_capa1, rain_col_map, df)

    if verbose:
        print(f"         Total Capa 1: {s_combined['cells_flagged']:,} celdas únicas  "
              f"({s_combined['pct_of_valid']:.2f}%)  "
              f"en {s_combined['stations_affected']:,} estaciones")

    # ── Guardar parquet ───────────────────────────────────────────────────────
    out_path = DATA_CATALOGS / "flags_capa1.parquet"
    flags_capa1.to_parquet(out_path)
    if verbose:
        print(f"\n[Guardado] {out_path}")

    # ── Figura resumen ────────────────────────────────────────────────────────
    flag_dict = {
        "zeros": f_zeros, "stuck": f_stuck,
        "negative": f_neg, "precision": f_prec, "combined": flags_capa1,
    }
    fig_path = _plot_capa1_summary(flag_dict, rain_cols, rain_col_map)
    if verbose:
        print(f"[Figura]  {fig_path.name}")

    # ── Tabla de resumen ──────────────────────────────────────────────────────
    summary = pd.DataFrame([s_zeros, s_stuck, s_neg, s_prec, s_combined])
    if verbose:
        print("\n── Resumen ──")
        print(summary.to_string(index=False))

        # Top-10 estaciones más flaggeadas
        top10 = (
            pd.DataFrame({
                "#Station": df["#Station"].values,
                "State": df["State"].values,
                "n_flags": flags_capa1.sum(axis=1).values,
                "pct_complete": df["pct_complete"].values,
            })
            .query("n_flags > 0")
            .sort_values("n_flags", ascending=False)
            .head(10)
        )
        print("\n── Top-10 estaciones más flaggeadas (Capa 1) ──")
        print(top10.to_string(index=False))

        # Columnas-mes con más flags
        top_months = (
            flags_capa1.sum(axis=0)
            .sort_values(ascending=False)
            .head(10)
        )
        print("\n── Top-10 meses con más flags ──")
        for col, cnt in top_months.items():
            m, y = rain_col_map[col]
            print(f"  {y}-{m:02d}: {cnt:,} estaciones flaggeadas")

    print("\n[OK] T2.1 completado.")
    return flags_capa1


# ═══════════════════════════════════════════════════════════════════════════════
# T2.2 — CAPA 2: Anomalías Univariadas Contextualizadas
# ═══════════════════════════════════════════════════════════════════════════════

def _monthly_groups(rain_col_map: dict) -> dict[int, list[str]]:
    """Devuelve {mes: [cols]} preservando inserción por año."""
    groups: dict[int, list[str]] = {}
    for col, (m, _) in rain_col_map.items():
        groups.setdefault(m, []).append(col)
    return groups


def zscore_seasonal(
    df: pd.DataFrame,
    rain_cols: list[str],
    rain_col_map: dict,
    threshold: float = 3.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    T2.2.1 — Z-score robusto por estación-mes.

    z = (x − mediana_mes) / MAD_mes   (MAD escalado a σ-normal)

    Computa por mes calendario (1–12) usando todos los años disponibles
    como referencia histórica de cada estación.

    Retorna (flags, scores) con mismas dimensiones que (estaciones × rain_cols).
    """
    from scipy.stats import median_abs_deviation as _mad

    mg = _monthly_groups(rain_col_map)
    col_pos = {col: i for i, col in enumerate(rain_cols)}
    n_st, n_rc = len(df), len(rain_cols)

    flags_arr  = np.zeros((n_st, n_rc), dtype=bool)
    scores_arr = np.full((n_st, n_rc), np.nan)

    for month, mcols in mg.items():
        data = df[mcols].values.astype(float)           # (n_st, n_yrs)

        # Mediana y MAD fila a fila, ignorando NaN
        medians = np.nanmedian(data, axis=1)             # (n_st,)
        with np.errstate(all="ignore"):
            mads = _mad(data, axis=1, scale="normal", nan_policy="omit")

        n_valid = np.sum(~np.isnan(data), axis=1)
        mads = np.where((n_valid < 3) | (mads == 0), np.nan, mads)

        with np.errstate(divide="ignore", invalid="ignore"):
            z = (data - medians[:, None]) / mads[:, None]  # (n_st, n_yrs)

        for j, col in enumerate(mcols):
            p = col_pos[col]
            scores_arr[:, p] = z[:, j]
            not_nan = ~np.isnan(z[:, j])
            flags_arr[not_nan, p] = np.abs(z[not_nan, j]) > threshold

    flags  = pd.DataFrame(flags_arr,  index=df.index, columns=rain_cols)
    scores = pd.DataFrame(scores_arr, index=df.index, columns=rain_cols)
    return flags, scores


def percentile_flags(
    df: pd.DataFrame,
    rain_cols: list[str],
    rain_col_map: dict,
    low: int = 2,
    high: int = 98,
) -> pd.DataFrame:
    """
    T2.2.2 — Percentiles condicionales por estación-mes.

    Flag si el valor está fuera del intervalo [P_low, P_high] del
    historial de la propia estación para ese mes calendario.
    """
    mg = _monthly_groups(rain_col_map)
    col_pos = {col: i for i, col in enumerate(rain_cols)}
    n_st, n_rc = len(df), len(rain_cols)
    flags_arr = np.zeros((n_st, n_rc), dtype=bool)

    for month, mcols in mg.items():
        data = df[mcols].values.astype(float)           # (n_st, n_yrs)

        n_valid = np.sum(~np.isnan(data), axis=1)
        enough  = n_valid >= 3                           # máscara (n_st,)

        with np.errstate(all="ignore"):
            p_lo = np.nanpercentile(data, low,  axis=1) # (n_st,)
            p_hi = np.nanpercentile(data, high, axis=1)

        for j, col in enumerate(mcols):
            p = col_pos[col]
            v  = data[:, j]
            ok = ~np.isnan(v) & enough
            flags_arr[ok, p] = (v[ok] < p_lo[ok]) | (v[ok] > p_hi[ok])

    return pd.DataFrame(flags_arr, index=df.index, columns=rain_cols)


def adjusted_boxplot_fences(data: np.ndarray) -> tuple[float, float]:
    """
    Fences del boxplot ajustado de Hubert & Vandervieren (2008).
    Corrige los límites de Tukey por la asimetría de la distribución
    usando el estadístico medcouple (MC).

    Retorna (lower_fence, upper_fence).
    """
    from statsmodels.stats.stattools import medcouple

    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    if iqr == 0:
        return float(q1), float(q3)
    mc = float(medcouple(data))
    if mc >= 0:
        lower = q1 - 1.5 * np.exp(-4 * mc) * iqr
        upper = q3 + 1.5 * np.exp( 3 * mc) * iqr
    else:
        lower = q1 - 1.5 * np.exp(-3 * mc) * iqr
        upper = q3 + 1.5 * np.exp( 4 * mc) * iqr
    return float(lower), float(upper)


def adjusted_boxplot_flags(
    df: pd.DataFrame,
    rain_cols: list[str],
    rain_col_map: dict,
    min_values: int = 10,
) -> pd.DataFrame:
    """
    T2.2.3 — Adjusted boxplot por cluster regional (Estado) y mes calendario.

    Agrupa todos los valores (estaciones × años) de un mismo estado y mes,
    calcula las fences ajustadas y flag los valores individuales que las superan.
    """
    mg = _monthly_groups(rain_col_map)
    col_pos = {col: i for i, col in enumerate(rain_cols)}
    n_st, n_rc = len(df), len(rain_cols)
    flags_arr = np.zeros((n_st, n_rc), dtype=bool)

    states = df["State"].unique()

    for state in states:
        state_mask = (df["State"] == state).values
        state_idx  = np.where(state_mask)[0]            # posiciones enteras globales

        for month, mcols in mg.items():
            state_data = df.iloc[state_idx][mcols].values.astype(float)
            # (n_state_stations, n_yrs)

            pooled = state_data.flatten()
            pooled = pooled[~np.isnan(pooled)]

            if len(pooled) < min_values:
                continue

            try:
                lower, upper = adjusted_boxplot_fences(pooled)
            except Exception:
                continue

            for j, col in enumerate(mcols):
                p = col_pos[col]
                v = state_data[:, j]
                valid = ~np.isnan(v)
                out   = valid & ((v < lower) | (v > upper))
                flags_arr[state_idx[out], p] = True

    return pd.DataFrame(flags_arr, index=df.index, columns=rain_cols)


# ── Figura resumen Capa 2 ─────────────────────────────────────────────────────

def _plot_capa2_summary(
    flag_dfs: dict[str, pd.DataFrame],
    scores: pd.DataFrame,
    rain_cols: list[str],
    rain_col_map: dict,
) -> Path:
    """Cuatro paneles: conteos, distribución z-scores, temporal, top meses."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    COLORS = {
        "zscore":    "#1565C0",
        "percentile":"#2E7D32",
        "adjboxplot":"#E65100",
        "combined":  "#212121",
    }
    methods = [m for m in flag_dfs if m != "combined"]

    # ── Panel 1: celdas flaggeadas por método ──
    counts = [int(flag_dfs[m].values.sum()) for m in methods + ["combined"]]
    labels = methods + ["combined"]
    bars = axes[0, 0].bar(
        labels, counts,
        color=[COLORS[m] for m in labels],
        edgecolor="white", width=0.6,
    )
    for bar, cnt in zip(bars, counts):
        axes[0, 0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts) * 0.01,
            f"{cnt:,}", ha="center", fontsize=8,
        )
    axes[0, 0].set_title("Flagged cells by method (T2.2)", fontsize=10)
    axes[0, 0].set_ylabel("No. of cells")
    axes[0, 0].tick_params(axis="x", rotation=15)

    # ── Panel 2: distribución de z-scores ──
    z_vals = scores.values.flatten()
    z_vals = z_vals[~np.isnan(z_vals)]
    axes[0, 1].hist(z_vals, bins=120, color="#1565C0", alpha=0.7,
                    edgecolor="none", density=True,
                    range=(np.percentile(z_vals, 0.5), np.percentile(z_vals, 99.5)))
    for thresh, ls in [(-3, "--"), (3, "--")]:
        axes[0, 1].axvline(thresh, color="red", linestyle=ls, linewidth=1.2,
                           label=f"|z|={abs(thresh)}")
    axes[0, 1].set_xlabel("Robust z-score")
    axes[0, 1].set_ylabel("Density")
    axes[0, 1].set_title(
        f"Z-score distribution (n={len(z_vals):,})\n"
        f"Flagged |z|>3: {(np.abs(z_vals)>3).sum():,} ({(np.abs(z_vals)>3).mean():.2%})",
        fontsize=10,
    )
    axes[0, 1].legend(fontsize=8)

    # ── Panel 3: distribución temporal (flags/mes) del combinado ──
    combined = flag_dfs["combined"]
    monthly_flags = combined.sum(axis=0).values
    year_ticks = {i: rain_col_map[c][1]
                  for i, c in enumerate(rain_cols) if rain_col_map[c][0] == 1}
    axes[1, 0].bar(range(len(rain_cols)), monthly_flags,
                   color=COLORS["combined"], alpha=0.75, width=1)
    axes[1, 0].set_xticks(list(year_ticks.keys()))
    axes[1, 0].set_xticklabels(list(year_ticks.values()), rotation=45, fontsize=8)
    axes[1, 0].set_title("Temporal distribution — Layer 2 flags combined", fontsize=10)
    axes[1, 0].set_ylabel("No. of flagged cells")
    axes[1, 0].set_xlabel("Month")

    # ── Panel 4: flags por estación (histograma) ──
    per_station = combined.sum(axis=1).values
    axes[1, 1].hist(per_station[per_station > 0], bins=40,
                    color=COLORS["combined"], alpha=0.8, edgecolor="white")
    axes[1, 1].set_title(
        f"Flags per station (affected only: n={int((per_station>0).sum()):,})",
        fontsize=10,
    )
    axes[1, 1].set_xlabel("No. of flagged months per station")
    axes[1, 1].set_ylabel("No. of stations")

    fig.suptitle("T2.2 — Layer 2 Summary: Contextualized Univariate Anomalies",
                 fontsize=12)
    fig.tight_layout()

    out = FIGURES / "anomaly_capa2_summary.png"
    fig.savefig(out, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return out


# ── Punto de entrada T2.2 ────────────────────────────────────────────────────

def run_t2_2(verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ejecuta T2.2 completo.

    Genera:
      - data/catalogs/flags_capa2.parquet   (bool, estaciones × rain_cols)
      - data/catalogs/zscores.parquet       (float, z-scores robustos)
      - outputs/figures/anomaly_capa2_summary.png

    Retorna (flags_capa2, scores).
    """
    from src.config import ZSCORE_THRESHOLD, PERCENTILE_LOW, PERCENTILE_HIGH

    DATA_CATALOGS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("=" * 60)
        print("T2.2 — Capa 2: Anomalías Univariadas Contextualizadas")
        print("=" * 60)

    df = pd.read_parquet(DATA_PROCESSED / "lluvia_clean.parquet")
    rain_col_map = parse_rain_columns(df.columns)
    rain_cols = _sorted_rain_cols(rain_col_map)

    if verbose:
        print(f"Datos: {df.shape[0]:,} estaciones × {len(rain_cols)} meses\n")

    # ── T2.2.1 ──────────────────────────────────────────────────────────────
    if verbose:
        print(f"[T2.2.1] Z-score estacional robusto (umbral |z| > {ZSCORE_THRESHOLD})...")
    f_z, scores = zscore_seasonal(df, rain_cols, rain_col_map,
                                  threshold=ZSCORE_THRESHOLD)
    n_z = int(f_z.values.sum())
    if verbose:
        z_vals = scores.values.flatten()
        z_vals = z_vals[~np.isnan(z_vals)]
        print(f"         {n_z:,} celdas  "
              f"({n_z / f_z.size * 100:.2f}%)  "
              f"en {int((f_z.sum(axis=1) > 0).sum()):,} estaciones")
        print(f"         z-scores: media={np.nanmean(np.abs(z_vals)):.2f}  "
              f"P99={np.nanpercentile(np.abs(z_vals), 99):.2f}  "
              f"max|z|={np.nanmax(np.abs(z_vals)):.2f}")

    # ── T2.2.2 ──────────────────────────────────────────────────────────────
    if verbose:
        print(f"\n[T2.2.2] Percentiles condicionales "
              f"(P{PERCENTILE_LOW}/P{PERCENTILE_HIGH} por estación-mes)...")
    f_pct = percentile_flags(df, rain_cols, rain_col_map,
                              low=PERCENTILE_LOW, high=PERCENTILE_HIGH)
    n_pct = int(f_pct.values.sum())
    if verbose:
        print(f"         {n_pct:,} celdas  "
              f"({n_pct / f_pct.size * 100:.2f}%)  "
              f"en {int((f_pct.sum(axis=1) > 0).sum()):,} estaciones")

    # ── T2.2.3 ──────────────────────────────────────────────────────────────
    if verbose:
        print("\n[T2.2.3] Adjusted boxplot (Hubert & Vandervieren) "
              "por Estado × mes...")
    f_adj = adjusted_boxplot_flags(df, rain_cols, rain_col_map, min_values=10)
    n_adj = int(f_adj.values.sum())
    if verbose:
        print(f"         {n_adj:,} celdas  "
              f"({n_adj / f_adj.size * 100:.2f}%)  "
              f"en {int((f_adj.sum(axis=1) > 0).sum()):,} estaciones")

    # ── Combinar (OR) ───────────────────────────────────────────────────────
    if verbose:
        print("\n[Combinando flags Capa 2 (OR)...]")
    flags_capa2 = f_z | f_pct | f_adj
    n_tot = int(flags_capa2.values.sum())
    if verbose:
        print(f"         Total Capa 2: {n_tot:,} celdas únicas  "
              f"({n_tot / flags_capa2.size * 100:.2f}%)  "
              f"en {int((flags_capa2.sum(axis=1) > 0).sum()):,} estaciones")

    # ── Guardar ─────────────────────────────────────────────────────────────
    flags_capa2.to_parquet(DATA_CATALOGS / "flags_capa2.parquet")
    scores.to_parquet(DATA_CATALOGS / "zscores.parquet")
    if verbose:
        print(f"\n[Guardado] flags_capa2.parquet")
        print(f"[Guardado] zscores.parquet")

    # ── Figura ──────────────────────────────────────────────────────────────
    fig_path = _plot_capa2_summary(
        {"zscore": f_z, "percentile": f_pct, "adjboxplot": f_adj,
         "combined": flags_capa2},
        scores, rain_cols, rain_col_map,
    )
    if verbose:
        print(f"[Figura]  {fig_path.name}")

    # ── Top flags ───────────────────────────────────────────────────────────
    if verbose:
        top10_st = (
            pd.DataFrame({
                "#Station": df["#Station"].values,
                "State": df["State"].values,
                "n_flags_c2": flags_capa2.sum(axis=1).values,
            })
            .query("n_flags_c2 > 0")
            .sort_values("n_flags_c2", ascending=False)
            .head(10)
        )
        print("\n── Top-10 estaciones más flaggeadas (Capa 2) ──")
        print(top10_st.to_string(index=False))

        top10_mo = (
            flags_capa2.sum(axis=0)
            .sort_values(ascending=False)
            .head(10)
        )
        print("\n── Top-10 meses con más flags ──")
        for col, cnt in top10_mo.items():
            m, y = rain_col_map[col]
            print(f"  {y}-{m:02d}: {int(cnt):,} estaciones")

    # Solape con Capa 1
    if (DATA_CATALOGS / "flags_capa1.parquet").exists():
        f1 = pd.read_parquet(DATA_CATALOGS / "flags_capa1.parquet")
        overlap = (f1 & flags_capa2).values.sum()
        if verbose:
            print(f"\n── Solapamiento Capa1 ∩ Capa2: {int(overlap):,} celdas "
                  f"(confirmadas por ambas capas)")

    print("\n[OK] T2.2 completado.")
    return flags_capa2, scores


# ═══════════════════════════════════════════════════════════════════════════════
# T2.3 — CAPA 3: Anomalías Espaciales
# ═══════════════════════════════════════════════════════════════════════════════

# Parámetros del semivariograma esférico ajustado en T1.3.5
_VARIO_PARAMS = {"psill": 6719.85, "range": 31.61, "nugget": 1440.91}


def kriging_residual_flags(
    df: pd.DataFrame,
    rain_cols: list[str],
    rain_col_map: dict,
    threshold_sigma: float = 2.5,
    min_stations: int = 50,
    cv_folds: int = 5,
    n_closest: int = 50,
) -> pd.DataFrame:
    """
    T2.3.1 — Residuos de kriging ordinario (5-fold LOO-CV).

    Para cada mes con ≥ min_stations estaciones válidas:
    - Divide las estaciones en cv_folds pliegues
    - Cada pliegue se predice por kriging ajustado en los otros pliegues
    - Residuo = observado − predicho; flag si |residuo| > threshold_sigma × σ

    Usa los parámetros del semivariograma esférico de T1.3.5 para evitar
    problemas de ajuste automático en meses con distribuciones atípicas.
    Usa ``n_closest`` vecinos en la predicción para evitar matrices singulares.

    Retorna DataFrame booleano (estaciones × rain_cols).
    """
    from pykrige.ok import OrdinaryKriging
    from sklearn.model_selection import KFold

    flags = pd.DataFrame(False, index=df.index, columns=rain_cols)

    coord_ok = df["Long"].notna() & df["Lat"].notna()

    for col in rain_cols:
        valid_mask = df[col].notna() & coord_ok
        n_valid = valid_mask.sum()
        if n_valid < min_stations:
            continue

        valid_idx = df.index[valid_mask]
        lons = df.loc[valid_mask, "Long"].values
        lats = df.loc[valid_mask, "Lat"].values
        vals = df.loc[valid_mask, col].values

        preds = np.full(n_valid, np.nan)
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

        for train_pos, test_pos in kf.split(np.arange(n_valid)):
            try:
                OK = OrdinaryKriging(
                    lons[train_pos], lats[train_pos], vals[train_pos],
                    variogram_model="spherical",
                    variogram_parameters=_VARIO_PARAMS,
                    verbose=False,
                    enable_plotting=False,
                )
                p, _ = OK.execute(
                    "points", lons[test_pos], lats[test_pos],
                    backend="loop", n_closest_points=min(n_closest, len(train_pos)),
                )
                preds[test_pos] = p.data
            except Exception:
                continue

        residuals = vals - preds
        valid_res = ~np.isnan(residuals)
        if valid_res.sum() < 10:
            continue

        sigma = residuals[valid_res].std()
        if sigma == 0:
            continue

        anomalous = valid_res & (np.abs(residuals) > threshold_sigma * sigma)
        flags.loc[valid_idx[anomalous], col] = True

    return flags


def lof_spatial_flags(
    df: pd.DataFrame,
    rain_cols: list[str],
    rain_col_map: dict,
    k: int = 20,
) -> pd.DataFrame:
    """
    T2.3.2 — LOF (Local Outlier Factor) espacial.

    Para cada mes, aplica LOF en el espacio (Long, Lat, precipitación)
    normalizado. Flag las estaciones con label = -1 (outlier).

    Retorna DataFrame booleano (estaciones × rain_cols).
    """
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.preprocessing import StandardScaler

    flags = pd.DataFrame(False, index=df.index, columns=rain_cols)
    coord_ok = df["Long"].notna() & df["Lat"].notna()

    for col in rain_cols:
        valid_mask = df[col].notna() & coord_ok
        if valid_mask.sum() < k + 1:
            continue

        valid_idx = df.index[valid_mask]
        X = df.loc[valid_mask, ["Lat", "Long", col]].values.astype(float)
        X = StandardScaler().fit_transform(X)

        try:
            lof = LocalOutlierFactor(n_neighbors=k, contamination="auto")
            labels = lof.fit_predict(X)
            flags.loc[valid_idx[labels == -1], col] = True
        except Exception:
            continue

    return flags


def lisa_flags(
    df: pd.DataFrame,
    rain_cols: list[str],
    rain_col_map: dict,
    k: int = 8,
    significance: float = 0.05,
    permutations: int = 199,
) -> pd.DataFrame:
    """
    T2.3.3 — LISA (Local Moran's I) con PySAL.

    Identifica estaciones High-Low (cuadrante 4) o Low-High (cuadrante 2)
    con p_sim < significance. Estas representan inconsistencias espaciales:
    valor alto rodeado de bajos, o bajo rodeado de altos.

    Retorna DataFrame booleano (estaciones × rain_cols).
    """
    from libpysal.weights import KNN as KNN_w
    from esda.moran import Moran_Local

    flags = pd.DataFrame(False, index=df.index, columns=rain_cols)
    coord_ok = df["Long"].notna() & df["Lat"].notna()

    for col in rain_cols:
        valid_mask = df[col].notna() & coord_ok
        if valid_mask.sum() < k + 1:
            continue

        valid_idx = df.index[valid_mask]
        coords = df.loc[valid_mask, ["Long", "Lat"]].values
        vals = df.loc[valid_mask, col].values

        try:
            w = KNN_w.from_array(coords, k=k)
            w.transform = "R"
            lisa = Moran_Local(vals, w, permutations=permutations, seed=42)
            spatial_anomaly = ((lisa.q == 2) | (lisa.q == 4)) & (lisa.p_sim < significance)
            flags.loc[valid_idx[spatial_anomaly], col] = True
        except Exception:
            continue

    return flags


# ── Figura resumen Capa 3 ─────────────────────────────────────────────────────

def _plot_capa3_summary(
    flag_dfs: dict[str, pd.DataFrame],
    rain_cols: list[str],
    rain_col_map: dict,
    df_meta: pd.DataFrame,
) -> Path:
    """Cuatro paneles: conteos, temporal, mapa espacial de flags, distribución por mes-cal."""
    import warnings
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    COLORS = {
        "kriging":  "#5C6BC0",
        "lof":      "#26A69A",
        "lisa":     "#EF5350",
        "combined": "#212121",
    }

    methods_order = ["kriging", "lof", "lisa", "combined"]
    labels_pretty = ["Kriging residuals", "LOF espacial", "LISA (HL/LH)", "Combinado"]

    # Panel 1: celdas flaggeadas por método
    counts = [int(flag_dfs[m].values.sum()) for m in methods_order]
    bars = axes[0, 0].bar(
        labels_pretty, counts,
        color=[COLORS[m] for m in methods_order],
        edgecolor="white", width=0.6,
    )
    for bar, cnt in zip(bars, counts):
        axes[0, 0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts) * 0.02,
            f"{cnt:,}", ha="center", fontsize=8,
        )
    axes[0, 0].set_title("Flagged cells by method (T2.3)", fontsize=10)
    axes[0, 0].set_ylabel("No. of cells")
    axes[0, 0].tick_params(axis="x", rotation=15)

    # Panel 2: distribución temporal del combinado
    combined = flag_dfs["combined"]
    monthly_flags = combined.sum(axis=0).values
    year_ticks = {i: rain_col_map[c][1]
                  for i, c in enumerate(rain_cols) if rain_col_map[c][0] == 1}
    axes[0, 1].bar(range(len(rain_cols)), monthly_flags,
                   color=COLORS["combined"], alpha=0.75, width=1)
    axes[0, 1].set_xticks(list(year_ticks.keys()))
    axes[0, 1].set_xticklabels(list(year_ticks.values()), rotation=45, fontsize=8)
    axes[0, 1].set_title("Temporal distribution — Layer 3 flags combined", fontsize=10)
    axes[0, 1].set_ylabel("No. of flagged cells")
    axes[0, 1].set_xlabel("Month")

    # Panel 3: mapa de frecuencia de flags por estación
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        per_station = combined.sum(axis=1)
        has_flag = per_station > 0
        sc = axes[1, 0].scatter(
            df_meta.loc[has_flag, "Long"],
            df_meta.loc[has_flag, "Lat"],
            c=per_station[has_flag],
            cmap="YlOrRd",
            s=15, alpha=0.8, vmin=1,
        )
        axes[1, 0].scatter(
            df_meta.loc[~has_flag, "Long"],
            df_meta.loc[~has_flag, "Lat"],
            color="lightgrey", s=3, alpha=0.4,
        )
    plt.colorbar(sc, ax=axes[1, 0], label="No. of flagged months")
    axes[1, 0].set_title("Spatial distribution of Layer 3 flags", fontsize=10)
    axes[1, 0].set_xlabel("Longitude")
    axes[1, 0].set_ylabel("Latitude")

    # Panel 4: flags por mes calendario (promedio de los 13 años)
    months_cal = [rain_col_map[c][0] for c in rain_cols]
    per_month_cal = {m: [] for m in range(1, 13)}
    for col, m in zip(rain_cols, months_cal):
        per_month_cal[m].append(int(combined[col].sum()))
    month_means = [np.mean(per_month_cal[m]) if per_month_cal[m] else 0
                   for m in range(1, 13)]
    MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    bars2 = axes[1, 1].bar(MONTH_NAMES, month_means,
                            color=[COLORS["combined"] if m in [5,6,7,8,9,10]
                                   else "#90A4AE" for m in range(1, 13)],
                            edgecolor="white")
    axes[1, 1].set_title("Average flags by calendar month\n(blue=dry, black=wet)", fontsize=10)
    axes[1, 1].set_ylabel("Average flagged stations")

    fig.suptitle("T2.3 — Layer 3 Summary: Spatial Anomalies", fontsize=12)
    fig.tight_layout()

    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / "anomaly_capa3_summary.png"
    fig.savefig(out, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return out


# ── Punto de entrada T2.3 ────────────────────────────────────────────────────

def run_t2_3(verbose: bool = True) -> pd.DataFrame:
    """
    Ejecuta T2.3 completo.

    Genera:
      - data/catalogs/flags_capa3.parquet   (bool, estaciones × rain_cols)
      - outputs/figures/anomaly_capa3_summary.png

    Retorna el DataFrame de flags combinado.
    """
    DATA_CATALOGS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("=" * 60)
        print("T2.3 — Capa 3: Anomalías Espaciales")
        print("=" * 60)

    df = pd.read_parquet(DATA_PROCESSED / "lluvia_clean.parquet")
    rain_col_map = parse_rain_columns(df.columns)
    rain_cols = _sorted_rain_cols(rain_col_map)

    if verbose:
        print(f"Datos: {df.shape[0]:,} estaciones × {len(rain_cols)} meses\n")

    # ── T2.3.1 — Kriging residuals ───────────────────────────────────────────
    if verbose:
        print("[T2.3.1] Residuos de kriging (5-fold CV, threshold=2.5σ)...")
    f_krig = kriging_residual_flags(df, rain_cols, rain_col_map,
                                    threshold_sigma=2.5, min_stations=50)
    n_krig = int(f_krig.values.sum())
    if verbose:
        print(f"         {n_krig:,} celdas  ({n_krig / f_krig.size * 100:.2f}%)  "
              f"en {int((f_krig.sum(axis=1) > 0).sum()):,} estaciones")

    # ── T2.3.2 — LOF espacial ────────────────────────────────────────────────
    if verbose:
        print("\n[T2.3.2] LOF espacial (k=20, espacio Lat×Long×Prec)...")
    from src.config import LOF_K_NEIGHBORS
    f_lof = lof_spatial_flags(df, rain_cols, rain_col_map, k=LOF_K_NEIGHBORS)
    n_lof = int(f_lof.values.sum())
    if verbose:
        print(f"         {n_lof:,} celdas  ({n_lof / f_lof.size * 100:.2f}%)  "
              f"en {int((f_lof.sum(axis=1) > 0).sum()):,} estaciones")

    # ── T2.3.3 — LISA ────────────────────────────────────────────────────────
    if verbose:
        print("\n[T2.3.3] LISA — Local Moran's I (k=8, α=0.05, 199 permutaciones)...")
    f_lisa = lisa_flags(df, rain_cols, rain_col_map,
                        k=8, significance=0.05, permutations=199)
    n_lisa = int(f_lisa.values.sum())
    if verbose:
        print(f"         {n_lisa:,} celdas  ({n_lisa / f_lisa.size * 100:.2f}%)  "
              f"en {int((f_lisa.sum(axis=1) > 0).sum()):,} estaciones")

    # ── Combinar (OR) ────────────────────────────────────────────────────────
    if verbose:
        print("\n[Combinando flags Capa 3 (OR)...]")
    flags_capa3 = f_krig | f_lof | f_lisa
    n_tot = int(flags_capa3.values.sum())
    n_sta = int((flags_capa3.sum(axis=1) > 0).sum())
    if verbose:
        print(f"         Total Capa 3: {n_tot:,} celdas únicas  "
              f"({n_tot / flags_capa3.size * 100:.2f}%)  "
              f"en {n_sta:,} estaciones")

    # ── Guardar ──────────────────────────────────────────────────────────────
    out_path = DATA_CATALOGS / "flags_capa3.parquet"
    flags_capa3.to_parquet(out_path)
    if verbose:
        print(f"\n[Guardado] {out_path}")

    # ── Figura ───────────────────────────────────────────────────────────────
    fig_path = _plot_capa3_summary(
        {"kriging": f_krig, "lof": f_lof, "lisa": f_lisa, "combined": flags_capa3},
        rain_cols, rain_col_map, df,
    )
    if verbose:
        print(f"[Figura]  {fig_path.name}")

    # ── Estadísticos finales ─────────────────────────────────────────────────
    if verbose:
        summary = pd.DataFrame([
            {"método": "kriging", "celdas": n_krig, "pct": n_krig/flags_capa3.size*100,
             "estaciones": int((f_krig.sum(axis=1)>0).sum())},
            {"método": "lof",     "celdas": n_lof,  "pct": n_lof/flags_capa3.size*100,
             "estaciones": int((f_lof.sum(axis=1)>0).sum())},
            {"método": "lisa",    "celdas": n_lisa, "pct": n_lisa/flags_capa3.size*100,
             "estaciones": int((f_lisa.sum(axis=1)>0).sum())},
            {"método": "combined","celdas": n_tot,  "pct": n_tot/flags_capa3.size*100,
             "estaciones": n_sta},
        ])
        print("\n── Resumen ──")
        print(summary.to_string(index=False))

        top10_st = (
            pd.DataFrame({
                "#Station": df["#Station"].values,
                "State": df["State"].values,
                "n_flags_c3": flags_capa3.sum(axis=1).values,
            })
            .query("n_flags_c3 > 0")
            .sort_values("n_flags_c3", ascending=False)
            .head(10)
        )
        print("\n── Top-10 estaciones más flaggeadas (Capa 3) ──")
        print(top10_st.to_string(index=False))

        # Solape con capas anteriores
        for layer, path in [("Capa1", "flags_capa1.parquet"),
                             ("Capa2", "flags_capa2.parquet")]:
            fpath = DATA_CATALOGS / path
            if fpath.exists():
                f_prev = pd.read_parquet(fpath)
                ov = int((f_prev & flags_capa3).values.sum())
                print(f"\n── Solapamiento {layer} ∩ Capa3: {ov:,} celdas")

    print("\n[OK] T2.3 completado.")
    return flags_capa3


# ═══════════════════════════════════════════════════════════════════════════════
# T2.4 — CAPA 4: Anomalías Multivariadas (Perfil Anual)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_profiles(
    df: pd.DataFrame,
    rain_col_map: dict,
    min_months: int = 12,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Construye perfiles anuales de 12 componentes (un valor por mes calendárico).

    Solo incluye estación-años con ≥ min_months meses válidos.
    Excluye 2026 (año incompleto).

    Retorna:
      profiles : np.ndarray (n_profiles, 12), valores crudos sin transformar
      meta     : DataFrame con columnas [station_idx, station_id, state, year]
    """
    years_full = sorted({y for _, (_, y) in rain_col_map.items() if y < 2026})

    year_cols: dict[int, list[str]] = {}
    for yr in years_full:
        year_cols[yr] = sorted(
            [c for c, (m, y) in rain_col_map.items() if y == yr],
            key=lambda c: rain_col_map[c][0],
        )

    profiles_list: list[np.ndarray] = []
    meta_list: list[dict] = []

    for i_st in df.index:
        row = df.loc[i_st]
        st_id = row["#Station"]
        state = row["State"]
        for yr, cols in year_cols.items():
            vals = row[cols].values.astype(float)
            if np.sum(~np.isnan(vals)) < min_months:
                continue
            profiles_list.append(vals)
            meta_list.append({
                "station_idx": i_st,
                "station_id": st_id,
                "state": state,
                "year": yr,
            })

    return np.array(profiles_list), pd.DataFrame(meta_list)


def _profiles_to_flags(
    meta: pd.DataFrame,
    anomalous_mask: np.ndarray,
    df: pd.DataFrame,
    rain_cols: list[str],
    rain_col_map: dict,
) -> pd.DataFrame:
    """
    Proyecta flags de nivel perfil (estación × año) al formato (estación × mes).

    Si el perfil (station_idx, year) está flaggeado, marca TRUE en todos
    los meses de ese año para esa estación.
    """
    flags = pd.DataFrame(False, index=df.index, columns=rain_cols)
    anom_meta = meta[anomalous_mask]
    if anom_meta.empty:
        return flags

    year_to_cols: dict[int, list[str]] = {}
    for yr in anom_meta["year"].unique():
        year_to_cols[yr] = [c for c, (_, y) in rain_col_map.items() if y == yr]

    for yr, yr_cols in year_to_cols.items():
        st_indices = anom_meta.loc[anom_meta["year"] == yr, "station_idx"].values
        flags.loc[st_indices, yr_cols] = True

    return flags


# ── T2.4.1 — Isolation Forest ─────────────────────────────────────────────────

def isolation_forest_profiles(
    profiles: np.ndarray,
    contamination: str | float = "auto",
    seed: int = 42,
) -> np.ndarray:
    """
    T2.4.1 — Isolation Forest sobre perfiles anuales log1p.

    Retorna máscara booleana (n_profiles,), True = anomalía.
    """
    from sklearn.ensemble import IsolationForest

    X = np.log1p(profiles)
    clf = IsolationForest(
        contamination=contamination,
        random_state=seed,
        n_estimators=300,
    )
    labels = clf.fit_predict(X)
    return labels == -1


# ── T2.4.2 — Autoencoder ──────────────────────────────────────────────────────

def _train_autoencoder(
    X: np.ndarray,
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    verbose: bool = True,
) -> tuple:
    """
    Entrena el autoencoder 12→6→3→6→12 sobre perfiles log1p.

    Retorna (model, mse_per_profile).
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    class RainfallAutoencoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(12, 6), nn.ReLU(),
                nn.Linear(6, 3),  nn.ReLU(),
            )
            self.decoder = nn.Sequential(
                nn.Linear(3, 6),  nn.ReLU(),
                nn.Linear(6, 12),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.decoder(self.encoder(x))

    device = torch.device("cpu")
    tensor = torch.tensor(X, dtype=torch.float32).to(device)
    loader = DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=True)

    model = RainfallAutoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for (batch,) in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch), batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch)
        if verbose and epoch % 50 == 0:
            print(f"         epoch {epoch}/{epochs}  loss={epoch_loss/len(X):.5f}")

    model.eval()
    with torch.no_grad():
        X_hat = model(tensor).cpu().numpy()
    mse = np.mean((X - X_hat) ** 2, axis=1)
    return model, mse


def autoencoder_flags(
    profiles: np.ndarray,
    threshold_pct: float = 95.0,
    epochs: int = 200,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    T2.4.2 — Autoencoder de reconstrucción sobre perfiles log1p.

    Anomalía = MSE(x, x̂) > percentil threshold_pct del MSE global.

    Retorna (máscara booleana, array de MSE por perfil).
    """
    X = np.log1p(profiles)
    _, mse = _train_autoencoder(X, epochs=epochs, verbose=verbose)
    threshold = np.percentile(mse, threshold_pct)
    return mse > threshold, mse


# ── T2.4.3 — Distancia de Mahalanobis robusta (MCD) ──────────────────────────

def mahalanobis_flags(
    profiles: np.ndarray,
    alpha: float = 0.025,
    min_profiles: int = 30,
) -> np.ndarray:
    """
    T2.4.3 — Mahalanobis robusta (MCD) aplicada perfil a perfil.

    Umbral: χ²(p=12, 1−α).  Retorna máscara booleana (n_profiles,).
    """
    from sklearn.covariance import MinCovDet
    from scipy.stats import chi2

    if len(profiles) < min_profiles:
        return np.zeros(len(profiles), dtype=bool)

    X = np.log1p(profiles)
    threshold = chi2.ppf(1 - alpha, df=X.shape[1])

    try:
        mcd = MinCovDet(random_state=42).fit(X)
        distances = mcd.mahalanobis(X)
    except Exception:
        return np.zeros(len(profiles), dtype=bool)

    return distances > threshold


def mahalanobis_flags_by_state(
    profiles: np.ndarray,
    meta: pd.DataFrame,
    alpha: float = 0.025,
    min_profiles: int = 30,
) -> np.ndarray:
    """
    T2.4.3 — MCD aplicada por estado (cluster regional).

    Para cada estado con ≥ min_profiles perfiles, ajusta MinCovDet
    independientemente y flag con umbral χ²(12, 1−α).
    """
    anomalous = np.zeros(len(profiles), dtype=bool)

    for state in meta["state"].unique():
        state_mask = (meta["state"] == state).values
        if state_mask.sum() < min_profiles:
            continue
        state_flags = mahalanobis_flags(
            profiles[state_mask], alpha=alpha, min_profiles=min_profiles,
        )
        anomalous[state_mask] = state_flags

    return anomalous


# ── Figura resumen Capa 4 ─────────────────────────────────────────────────────

def _plot_capa4_summary(
    flag_dfs: dict[str, pd.DataFrame],
    mse_values: np.ndarray,
    mse_threshold: float,
    rain_cols: list[str],
    rain_col_map: dict,
    df_meta: pd.DataFrame,
) -> Path:
    """Cuatro paneles: conteos por método, MSE del autoencoder, temporal, por año."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    COLORS = {
        "isoforest": "#7B1FA2",
        "autoencoder": "#0288D1",
        "mahalanobis": "#388E3C",
        "combined": "#212121",
    }
    methods_order = ["isoforest", "autoencoder", "mahalanobis", "combined"]
    labels_pretty = ["Isolation Forest", "Autoencoder", "Mahalanobis (MCD)", "Combinado"]

    # Panel 1: celdas flaggeadas por método
    counts = [int(flag_dfs[m].values.sum()) for m in methods_order]
    bars = axes[0, 0].bar(
        labels_pretty, counts,
        color=[COLORS[m] for m in methods_order],
        edgecolor="white", width=0.6,
    )
    for bar, cnt in zip(bars, counts):
        axes[0, 0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts) * 0.02,
            f"{cnt:,}", ha="center", fontsize=8,
        )
    axes[0, 0].set_title("Flagged cells by method (T2.4)", fontsize=10)
    axes[0, 0].set_ylabel("No. of cells")
    axes[0, 0].tick_params(axis="x", rotation=15)

    # Panel 2: distribución de MSE del autoencoder
    axes[0, 1].hist(mse_values, bins=80, color=COLORS["autoencoder"],
                    alpha=0.75, edgecolor="none",
                    range=(0, np.percentile(mse_values, 99.5)))
    axes[0, 1].axvline(mse_threshold, color="red", linestyle="--", linewidth=1.5,
                       label=f"P95 = {mse_threshold:.4f}")
    axes[0, 1].set_xlabel("Reconstruction MSE (log1p scale)")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].set_title(
        f"MSE distribution — Autoencoder\n"
        f"Anomalies (MSE > P95): {(mse_values > mse_threshold).sum():,} profiles",
        fontsize=10,
    )
    axes[0, 1].legend(fontsize=9)

    # Panel 3: flags por año (perfiles por año flaggeados)
    combined = flag_dfs["combined"]
    year_ticks = {i: rain_col_map[c][1]
                  for i, c in enumerate(rain_cols) if rain_col_map[c][0] == 1}
    monthly_flags = combined.sum(axis=0).values
    axes[1, 0].bar(range(len(rain_cols)), monthly_flags,
                   color=COLORS["combined"], alpha=0.75, width=1)
    axes[1, 0].set_xticks(list(year_ticks.keys()))
    axes[1, 0].set_xticklabels(list(year_ticks.values()), rotation=45, fontsize=8)
    axes[1, 0].set_title("Temporal distribution — Layer 4 flags combined", fontsize=10)
    axes[1, 0].set_ylabel("No. of flagged cells")
    axes[1, 0].set_xlabel("Month")

    # Panel 4: flags por estado (top-20)
    per_state = (
        combined.any(axis=1)
        .groupby(df_meta["State"].values)
        .sum()
        .sort_values(ascending=False)
        .head(20)
    )
    axes[1, 1].barh(per_state.index[::-1], per_state.values[::-1],
                    color=COLORS["combined"], alpha=0.75)
    axes[1, 1].set_title("Stations affected by state (top-20)", fontsize=10)
    axes[1, 1].set_xlabel("No. of stations with ≥1 flagged month")

    fig.suptitle("T2.4 — Layer 4 Summary: Multivariate Anomalies (Annual Profile)",
                 fontsize=12)
    fig.tight_layout()

    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / "anomaly_capa4_summary.png"
    fig.savefig(out, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return out


# ── Punto de entrada T2.4 ────────────────────────────────────────────────────

def run_t2_4(verbose: bool = True) -> pd.DataFrame:
    """
    Ejecuta T2.4 completo.

    Genera:
      - data/catalogs/flags_capa4.parquet   (bool, estaciones × rain_cols)
      - outputs/figures/anomaly_capa4_summary.png

    Retorna el DataFrame de flags combinado.
    """
    from src.config import MAHAL_CHI2_ALPHA, ISOLATION_FOREST_CONTAMINATION, SEED

    DATA_CATALOGS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("=" * 60)
        print("T2.4 — Capa 4: Anomalías Multivariadas (Perfil Anual)")
        print("=" * 60)

    df = pd.read_parquet(DATA_PROCESSED / "lluvia_clean.parquet")
    rain_col_map = parse_rain_columns(df.columns)
    rain_cols = _sorted_rain_cols(rain_col_map)

    # ── Construir perfiles ──────────────────────────────────────────────────
    if verbose:
        print("\n[Construyendo perfiles anuales (12 meses completos, 2013–2025)...]")
    profiles, meta = _build_profiles(df, rain_col_map, min_months=12)
    n_profiles = len(profiles)
    if verbose:
        print(f"  {n_profiles:,} perfiles  ({meta['state'].nunique()} estados  "
              f"{meta['year'].nunique()} años)")

    # ── T2.4.1 — Isolation Forest ───────────────────────────────────────────
    if verbose:
        print(f"\n[T2.4.1] Isolation Forest (n_estimators=300, contamination={ISOLATION_FOREST_CONTAMINATION})...")
    mask_if = isolation_forest_profiles(
        profiles, contamination=ISOLATION_FOREST_CONTAMINATION, seed=SEED,
    )
    f_if = _profiles_to_flags(meta, mask_if, df, rain_cols, rain_col_map)
    n_if = int(f_if.values.sum())
    if verbose:
        print(f"         {mask_if.sum():,} perfiles anomalous ({mask_if.mean()*100:.1f}%)  "
              f"→ {n_if:,} celdas ({n_if/f_if.size*100:.2f}%)  "
              f"en {int((f_if.sum(axis=1)>0).sum()):,} estaciones")

    # ── T2.4.2 — Autoencoder ────────────────────────────────────────────────
    if verbose:
        print("\n[T2.4.2] Autoencoder 12→6→3→6→12 (200 épocas, log1p)...")
    mask_ae, mse_vals = autoencoder_flags(profiles, threshold_pct=95.0,
                                          epochs=200, verbose=verbose)
    mse_thr = float(np.percentile(mse_vals, 95))
    f_ae = _profiles_to_flags(meta, mask_ae, df, rain_cols, rain_col_map)
    n_ae = int(f_ae.values.sum())
    if verbose:
        print(f"         {mask_ae.sum():,} perfiles anomalous ({mask_ae.mean()*100:.1f}%)  "
              f"→ {n_ae:,} celdas ({n_ae/f_ae.size*100:.2f}%)  "
              f"en {int((f_ae.sum(axis=1)>0).sum()):,} estaciones")
        print(f"         MSE umbral (P95) = {mse_thr:.5f}")

    # ── T2.4.3 — Mahalanobis (MCD) por estado ───────────────────────────────
    if verbose:
        print(f"\n[T2.4.3] Mahalanobis robusta MCD por estado (α={MAHAL_CHI2_ALPHA})...")
    mask_mh = mahalanobis_flags_by_state(
        profiles, meta, alpha=MAHAL_CHI2_ALPHA, min_profiles=30,
    )
    f_mh = _profiles_to_flags(meta, mask_mh, df, rain_cols, rain_col_map)
    n_mh = int(f_mh.values.sum())
    if verbose:
        print(f"         {mask_mh.sum():,} perfiles anomalous ({mask_mh.mean()*100:.1f}%)  "
              f"→ {n_mh:,} celdas ({n_mh/f_mh.size*100:.2f}%)  "
              f"en {int((f_mh.sum(axis=1)>0).sum()):,} estaciones")

    # ── Combinar (OR) ────────────────────────────────────────────────────────
    if verbose:
        print("\n[Combinando flags Capa 4 (OR)...]")
    flags_capa4 = f_if | f_ae | f_mh
    n_tot = int(flags_capa4.values.sum())
    n_sta = int((flags_capa4.sum(axis=1) > 0).sum())
    if verbose:
        print(f"         Total Capa 4: {n_tot:,} celdas ({n_tot/flags_capa4.size*100:.2f}%)  "
              f"en {n_sta:,} estaciones")

    # ── Guardar ──────────────────────────────────────────────────────────────
    out_path = DATA_CATALOGS / "flags_capa4.parquet"
    flags_capa4.to_parquet(out_path)
    if verbose:
        print(f"\n[Guardado] {out_path}")

    # ── Figura ───────────────────────────────────────────────────────────────
    fig_path = _plot_capa4_summary(
        {"isoforest": f_if, "autoencoder": f_ae,
         "mahalanobis": f_mh, "combined": flags_capa4},
        mse_vals, mse_thr, rain_cols, rain_col_map, df,
    )
    if verbose:
        print(f"[Figura]  {fig_path.name}")

    # ── Resumen ───────────────────────────────────────────────────────────────
    if verbose:
        summary = pd.DataFrame([
            {"método": "isoforest",   "perfiles_anom": int(mask_if.sum()),
             "celdas": n_if, "pct": n_if/flags_capa4.size*100,
             "estaciones": int((f_if.sum(axis=1)>0).sum())},
            {"método": "autoencoder", "perfiles_anom": int(mask_ae.sum()),
             "celdas": n_ae, "pct": n_ae/flags_capa4.size*100,
             "estaciones": int((f_ae.sum(axis=1)>0).sum())},
            {"método": "mahalanobis", "perfiles_anom": int(mask_mh.sum()),
             "celdas": n_mh, "pct": n_mh/flags_capa4.size*100,
             "estaciones": int((f_mh.sum(axis=1)>0).sum())},
            {"método": "combined",    "perfiles_anom": int((mask_if|mask_ae|mask_mh).sum()),
             "celdas": n_tot, "pct": n_tot/flags_capa4.size*100,
             "estaciones": n_sta},
        ])
        print("\n── Resumen ──")
        print(summary.to_string(index=False))

        top10_st = (
            pd.DataFrame({
                "#Station": df["#Station"].values,
                "State": df["State"].values,
                "n_flags_c4": flags_capa4.sum(axis=1).values,
            })
            .query("n_flags_c4 > 0")
            .sort_values("n_flags_c4", ascending=False)
            .head(10)
        )
        print("\n── Top-10 estaciones más flaggeadas (Capa 4) ──")
        print(top10_st.to_string(index=False))

        for layer, path in [("Capa1","flags_capa1.parquet"),
                             ("Capa2","flags_capa2.parquet"),
                             ("Capa3","flags_capa3.parquet")]:
            fpath = DATA_CATALOGS / path
            if fpath.exists():
                f_prev = pd.read_parquet(fpath)
                ov = int((f_prev & flags_capa4).values.sum())
                print(f"\n── Solapamiento {layer} ∩ Capa4: {ov:,} celdas")

    print("\n[OK] T2.4 completado.")
    return flags_capa4


if __name__ == "__main__":
    run_t2_4(verbose=True)
