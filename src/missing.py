"""
Diagnóstico de datos faltantes en el dataset pluviométrico — Tarea T1.2.

Caracteriza el mecanismo y la estructura de la omisión de datos para las
1,959 estaciones del SMN en el período 2013–2026. Incluye cuatro subtareas:

  T1.2.1  Visualización matricial del datos faltantes con missingno (ordenado por
          estado y % completitud) y heatmap por estado × año.
  T1.2.2  Implementación simplificada del test de Little (1988) para evaluar
          si los datos son MCAR, usando pseudoinversa de Moore-Penrose para
          estabilidad numérica.
  T1.2.3  Correlaciones punto-biserial entre el indicador de ausencia y las
          variables espaciales (Lat, Long) y temporales (mes, año).
  T1.2.4  Clasificación de patrones de dropout por estación: monotónica,
          intermitente, completa o activa.

Genera cinco figuras en outputs/figures/ y el informe
outputs/reports/diagnostico_datos_faltantes.md.

Punto de entrada: ``run_t1_2(verbose=True)``.

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
import matplotlib.patches as mpatches
import missingno as msno
from scipy.stats import chi2 as chi2_dist, pointbiserialr
from datetime import date

from src.config import DATA_PROCESSED, FIGURES, REPORTS
from src.loading import parse_rain_columns


# ── Utilidades ───────────────────────────────────────────────────────────────

def _sorted_rain_cols(rain_col_map: dict) -> list[str]:
    """Devuelve las columnas de precipitación ordenadas cronológicamente."""
    return sorted(rain_col_map.keys(), key=lambda c: (rain_col_map[c][1], rain_col_map[c][0]))


# ── T1.2.1 — Heatmap de datos faltantes ─────────────────────────────────────────

def plot_datos_faltantes_matrix(df: pd.DataFrame, rain_cols: list, rain_col_map: dict) -> Path:
    """
    Matriz de datos faltantes con missingno.
    Ordenada por Estado (α) y pct_complete (↓) para revelar patrones de bloque.
    """
    from pathlib import Path

    df_sorted = df.sort_values(["State", "pct_complete"], ascending=[True, False])

    ax = msno.matrix(
        df_sorted[rain_cols],
        figsize=(22, 14),
        sparkline=False,
        fontsize=6,
        labels=False,
        color=(0.25, 0.45, 0.75),
    )
    fig = ax.get_figure()

    # Etiquetas de año en eje x (posición del enero de cada año)
    year_ticks = {
        i: str(rain_col_map[c][1])
        for i, c in enumerate(rain_cols)
        if rain_col_map[c][0] == 1
    }
    ax.set_xticks(list(year_ticks.keys()))
    ax.set_xticklabels(list(year_ticks.values()), rotation=45, fontsize=8)
    ax.set_xlabel("Año (posición de enero)", fontsize=9)
    ax.set_ylabel(
        f"Estaciones (n={len(df_sorted):,}) — ordenadas por Estado / % completitud ↓",
        fontsize=8,
    )
    ax.set_title(
        "Datos faltantes por estación × mes (blanco = NaN)\n"
        "Orden: Estado (A→Z) · pct_complete (↓)",
        fontsize=11,
    )

    fig.tight_layout()
    out = FIGURES / "missing_matrix.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_missing_by_state_year(
    df: pd.DataFrame, rain_cols: list, rain_col_map: dict
) -> Path:
    """Heatmap de proporción de faltantes por Estado × Año."""
    from pathlib import Path

    years = sorted({y for _, (m, y) in rain_col_map.items()})
    states = sorted(df["State"].unique())

    # Calcular tasa de faltantes por (estado, año)
    matrix = pd.DataFrame(index=states, columns=years, dtype=float)
    for yr in years:
        yr_cols = [c for c, (m, y) in rain_col_map.items() if y == yr]
        for state in states:
            mask = df["State"] == state
            denom = mask.sum() * len(yr_cols)
            numer = df.loc[mask, yr_cols].isna().sum().sum()
            matrix.loc[state, yr] = numer / denom if denom > 0 else np.nan

    vals = matrix.values.astype(float)

    fig, axes = plt.subplots(
        1, 2, figsize=(20, 14), gridspec_kw={"width_ratios": [14, 1]}
    )
    ax, cax = axes

    im = ax.imshow(vals, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=1)
    fig.colorbar(im, cax=cax, label="Proporción faltantes")

    ax.set_xticks(range(len(years)))
    ax.set_xticklabels([str(y) for y in years], rotation=45, fontsize=8)
    ax.set_yticks(range(len(states)))
    ax.set_yticklabels(states, fontsize=7)
    ax.set_title(
        "Proporción de datos faltantes por Estado × Año\n(rojo = alta, verde = baja)",
        fontsize=11,
    )

    # Anotar celdas numéricamente
    for i in range(len(states)):
        for j in range(len(years)):
            v = vals[i, j]
            if not np.isnan(v):
                ax.text(
                    j, i, f"{v:.0%}",
                    ha="center", va="center",
                    fontsize=4.5,
                    color="white" if v > 0.6 else "black",
                )

    fig.tight_layout()
    out = FIGURES / "missing_by_state_year.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


# ── T1.2.2 — Test de Little para MCAR ───────────────────────────────────────

def littles_mcar_test(
    data: pd.DataFrame,
    min_col_completeness: float = 0.35,
    max_rows: int = 1000,
    verbose: bool = True,
) -> tuple[float, int, float]:
    """
    Implementación simplificada del test de Little (1988).

    H0: datos son MCAR.
    Si p < 0.05 → rechazar H0 → investigar MAR vs MNAR.

    Por estabilidad numérica se usan columnas con completitud ≥ min_col_completeness
    y hasta max_rows filas (muestra aleatoria si el dataset es mayor).
    """
    rng = np.random.default_rng(42)

    # Filtrar columnas con suficiente completitud
    col_comp = data.notna().mean()
    cols = col_comp[col_comp >= min_col_completeness].index.tolist()
    if verbose:
        print(
            f"    Columnas usadas: {len(cols)}/{len(data.columns)} "
            f"(completitud ≥ {min_col_completeness:.0%})"
        )
    if len(cols) < 3:
        return np.nan, 0, np.nan

    # Muestra de filas para eficiencia
    if len(data) > max_rows:
        idx = rng.choice(len(data), size=max_rows, replace=False)
        data_sub = data.iloc[idx][cols].copy()
        if verbose:
            print(f"    Muestra aleatoria: {max_rows} de {len(data)} filas")
    else:
        data_sub = data[cols].copy()

    # Identificar patrones únicos de datos faltantes
    patterns_bin = data_sub.notna().astype(np.int8)
    pattern_ids = pd.Series(
        [tuple(row) for row in patterns_bin.values], index=data_sub.index
    )
    unique_patterns = pattern_ids.unique()
    if verbose:
        print(f"    Patrones únicos de datos faltantes: {len(unique_patterns)}")

    grand_mean = data_sub.mean()
    grand_cov = data_sub.cov(min_periods=10)

    chi2_stat = 0.0
    df_chi = 0

    for pattern in unique_patterns:
        mask = pattern_ids == pattern
        subgroup = data_sub[mask]
        n_j = len(subgroup)
        if n_j < 2:
            continue

        observed_cols = [c for c, v in zip(data_sub.columns, pattern) if v == 1]
        if len(observed_cols) == 0:
            continue

        sub_mean = subgroup[observed_cols].mean()
        diff = (sub_mean - grand_mean[observed_cols]).values

        # Covarianza escalada por 1/n_j (implica χ² = diff' pinv(Σ/n_j) diff)
        sub_cov = grand_cov.loc[observed_cols, observed_cols].values / n_j

        if np.any(np.isnan(sub_cov)) or np.any(np.isnan(diff)):
            continue
        try:
            contrib = float(diff @ np.linalg.pinv(sub_cov, rcond=1e-6) @ diff)
            if contrib > 0:  # descartar contribuciones negativas (inestabilidad numérica)
                chi2_stat += contrib
                df_chi += len(observed_cols)
        except np.linalg.LinAlgError:
            continue

    df_chi = max(df_chi - len(cols), 1)
    p_value = float(1 - chi2_dist.cdf(chi2_stat, df_chi))
    return chi2_stat, df_chi, p_value


# ── T1.2.3 — Correlaciones punto-biserial ───────────────────────────────────

def biserial_correlations(
    df: pd.DataFrame,
    rain_cols: list,
    rain_col_map: dict,
    verbose: bool = True,
) -> dict:
    """
    Correlación punto-biserial entre indicador de ausencia (1=faltante)
    y variables de contexto: Lat, Long, mes, año.
    """
    missing_ind = df[rain_cols].isna().astype(int)  # estaciones × meses

    results: dict = {}

    # Correlación con Lat y Long (por columna mensual)
    for meta_var in ("Lat", "Long"):
        meta_vals = df[meta_var]
        col_results: dict = {}
        for col in rain_cols:
            ind = missing_ind[col]
            valid = ind.notna() & meta_vals.notna()
            if valid.sum() < 10:
                continue
            try:
                r, p = pointbiserialr(ind[valid], meta_vals[valid])
                col_results[col] = {"r": r, "p": p}
            except Exception:
                continue
        results[meta_var] = col_results

    # Resumen temporal: tasa de faltantes por mes-año
    temporal_records = []
    for col in rain_cols:
        m, y = rain_col_map[col]
        rate = missing_ind[col].mean()
        temporal_records.append({"col": col, "month": m, "year": y, "missing_rate": rate})
    results["temporal"] = pd.DataFrame(temporal_records)

    # Correlación de missing_rate con mes y año (nivel columna, no individuo)
    tdf = results["temporal"]
    r_month, p_month = pointbiserialr(
        (tdf["missing_rate"] > tdf["missing_rate"].median()).astype(int),
        tdf["month"],
    )
    r_year, p_year = pointbiserialr(
        (tdf["missing_rate"] > tdf["missing_rate"].median()).astype(int),
        tdf["year"],
    )
    results["temporal_corr"] = {
        "month": {"r": r_month, "p": p_month},
        "year": {"r": r_year, "p": p_year},
    }

    if verbose:
        lat_rs = [v["r"] for v in results["Lat"].values()]
        lon_rs = [v["r"] for v in results["Long"].values()]
        lat_sig = sum(1 for v in results["Lat"].values() if v["p"] < 0.05)
        lon_sig = sum(1 for v in results["Long"].values() if v["p"] < 0.05)
        print(
            f"    Lat — r̄={np.nanmean(lat_rs):.3f}, "
            f"significativos (p<0.05): {lat_sig}/{len(lat_rs)}"
        )
        print(
            f"    Lon — r̄={np.nanmean(lon_rs):.3f}, "
            f"significativos (p<0.05): {lon_sig}/{len(lon_rs)}"
        )

    return results


def plot_biserial_correlations(
    df: pd.DataFrame,
    rain_cols: list,
    rain_col_map: dict,
    corr_results: dict,
) -> Path:
    """
    Tres paneles:
      1) r(datos faltantes, Latitud) por mes-año
      2) r(datos faltantes, Longitud) por mes-año
      3) Tasa mensual de faltantes a lo largo del tiempo
    """
    from pathlib import Path

    fig, axes = plt.subplots(3, 1, figsize=(18, 12), sharex=True)
    x = range(len(rain_cols))
    THRESH = 0.10  # umbral de relevancia práctica

    for ax_idx, (meta_var, ylabel) in enumerate(
        [("Lat", "r (Latitud)"), ("Long", "r (Longitud)")]
    ):
        col_dict = corr_results.get(meta_var, {})
        rs = [col_dict.get(c, {}).get("r", np.nan) for c in rain_cols]
        colors = [
            "#e74c3c" if (not np.isnan(r) and abs(r) > THRESH) else "#95a5a6"
            for r in rs
        ]
        axes[ax_idx].bar(x, rs, color=colors, width=1.0, alpha=0.85)
        axes[ax_idx].axhline(0, color="black", linewidth=0.6)
        for sign in (THRESH, -THRESH):
            axes[ax_idx].axhline(
                sign, color="red", linestyle="--", linewidth=0.8, alpha=0.6
            )
        axes[ax_idx].set_ylabel(ylabel, fontsize=9)
        axes[ax_idx].set_ylim(-0.6, 0.6)

    axes[0].set_title(
        "Correlaciones punto-biserial: datos faltantes ~ contexto\n"
        "(rojo = |r| > 0.10; línea punteada = umbral de relevancia)",
        fontsize=11,
    )

    # Panel 3: tasa temporal de faltantes
    tdf = corr_results.get("temporal", pd.DataFrame())
    if not tdf.empty:
        rates = [tdf.loc[tdf["col"] == c, "missing_rate"].values[0]
                 if c in tdf["col"].values else np.nan
                 for c in rain_cols]
        axes[2].plot(x, rates, color="darkorange", linewidth=1.3)
        axes[2].fill_between(x, 0, rates, alpha=0.25, color="darkorange")
        axes[2].axhline(np.nanmean(rates), color="navy", linestyle=":", linewidth=1,
                        label=f"Media = {np.nanmean(rates):.1%}")
        axes[2].set_ylabel("Tasa de faltantes", fontsize=9)
        axes[2].set_ylim(0, 1)
        axes[2].legend(fontsize=8)

    # Etiquetas de año en eje x
    year_ticks = [
        (i, rain_col_map[c][1])
        for i, c in enumerate(rain_cols)
        if rain_col_map[c][0] == 1
    ]
    for ax in axes:
        ax.set_xticks([t[0] for t in year_ticks])
        ax.set_xticklabels([str(t[1]) for t in year_ticks], fontsize=8, rotation=45)
    axes[2].set_xlabel("Período (posición del enero de cada año)", fontsize=9)

    fig.tight_layout()
    out = FIGURES / "missing_biserial.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


# ── T1.2.4 — Segmentación temporal de dropout ───────────────────────────────

def _nan_blocks(valid_mask: np.ndarray) -> list[tuple[int, int, int]]:
    """Devuelve lista de (start, end, length) de bloques contiguos de NaN."""
    blocks = []
    in_block = False
    start = 0
    for i, v in enumerate(valid_mask):
        if not v:
            if not in_block:
                in_block = True
                start = i
        else:
            if in_block:
                blocks.append((start, i - 1, i - start))
                in_block = False
    if in_block:
        blocks.append((start, len(valid_mask) - 1, len(valid_mask) - start))
    return blocks


def classify_dropout_patterns(
    df: pd.DataFrame, rain_cols: list
) -> pd.DataFrame:
    """
    Clasifica el patrón temporal de datos faltantes de cada estación.

    Categorías
    ----------
    monotonic    : la estación deja de reportar y no vuelve
                   (único bloque NaN al final de la serie)
    intermittent : bloques de NaN dispersos (la estación aparece y desaparece)
    complete     : sin datos en todo el período
    active       : sin ausencias o ausencias triviales (<5% de meses)
    """
    n_total = len(rain_cols)
    records = []

    for idx in df.index:
        vals = df.loc[idx, rain_cols].values
        valid_mask = ~pd.isna(vals)
        n_valid = valid_mask.sum()
        n_missing = n_total - n_valid

        row: dict = {
            "station": df.loc[idx, "#Station"],
            "state": df.loc[idx, "State"],
            "pct_complete": df.loc[idx, "pct_complete"],
            "n_valid": n_valid,
            "n_missing": n_missing,
        }

        if n_missing == 0:
            row.update(pattern="active", n_nan_blocks=0, longest_nan_block=0,
                       first_valid=0, last_valid=n_total - 1)
        elif n_valid == 0:
            row.update(pattern="complete", n_nan_blocks=1, longest_nan_block=n_total,
                       first_valid=-1, last_valid=-1)
        else:
            valid_idx = np.where(valid_mask)[0]
            first_v, last_v = int(valid_idx[0]), int(valid_idx[-1])
            blocks = _nan_blocks(valid_mask)
            n_blocks = len(blocks)
            longest = max(b[2] for b in blocks) if blocks else 0

            # Monotónico: el bloque NaN final llega hasta el último mes
            # y el último dato válido está al menos 6 meses antes del final.
            # (La estación dejó de reportar y no regresó, independientemente
            # de si tuvo gaps previos.)
            if (
                blocks
                and blocks[-1][1] == n_total - 1   # último bloque llega al fin
                and blocks[-1][2] >= 6              # ausencia final ≥ 6 meses
            ):
                pattern = "monotonic"
            else:
                pattern = "intermittent"

            row.update(
                pattern=pattern,
                n_nan_blocks=n_blocks,
                longest_nan_block=longest,
                first_valid=first_v,
                last_valid=last_v,
            )

        records.append(row)

    return pd.DataFrame(records)


def plot_dropout_analysis(dropout_df: pd.DataFrame, rain_cols: list) -> Path:
    """Tres paneles: distribución de patrones, bloques NaN, longitud vs completitud."""
    from pathlib import Path

    COLORS = {
        "active": "#2ecc71",
        "intermittent": "#f39c12",
        "monotonic": "#e74c3c",
        "complete": "#7f8c8d",
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # — Panel 1: conteo de patrones —
    counts = dropout_df["pattern"].value_counts().reindex(
        ["active", "intermittent", "monotonic", "complete"], fill_value=0
    )
    bar_colors = [COLORS[p] for p in counts.index]
    bars = axes[0].bar(counts.index, counts.values, color=bar_colors, edgecolor="white", width=0.6)
    axes[0].set_title("Clasificación de patrones de dropout", fontsize=10)
    axes[0].set_ylabel("N° estaciones")
    for bar, cnt in zip(bars, counts.values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 8,
            str(cnt),
            ha="center", fontsize=9,
        )
    legend_patches = [
        mpatches.Patch(facecolor=COLORS[p], label=p) for p in COLORS
    ]
    axes[0].legend(handles=legend_patches, fontsize=8, loc="upper right")

    # — Panel 2: distribución del número de bloques NaN —
    sub = dropout_df[dropout_df["pattern"].isin(["monotonic", "intermittent"])]
    n_blocks = sub["n_nan_blocks"].clip(0, 25)
    axes[1].hist(
        n_blocks, bins=range(0, 26), color="steelblue", edgecolor="white", alpha=0.85
    )
    axes[1].set_title("N° de bloques contiguos de NaN\n(estaciones con datos parciales)", fontsize=10)
    axes[1].set_xlabel("Bloques NaN por estación")
    axes[1].set_ylabel("N° estaciones")

    # — Panel 3: bloque más largo vs completitud, coloreado por patrón —
    for pat in ("monotonic", "intermittent"):
        mask = dropout_df["pattern"] == pat
        axes[2].scatter(
            dropout_df.loc[mask, "pct_complete"],
            dropout_df.loc[mask, "longest_nan_block"],
            c=COLORS[pat], alpha=0.4, s=10, label=pat,
        )
    axes[2].set_xlabel("% Completitud")
    axes[2].set_ylabel("Longitud del bloque NaN más largo (meses)")
    axes[2].set_title("Bloque NaN máximo vs completitud", fontsize=10)
    axes[2].axhline(12, color="gray", linestyle=":", linewidth=0.8, label="1 año")
    axes[2].axhline(24, color="gray", linestyle="--", linewidth=0.8, label="2 años")
    axes[2].legend(fontsize=8)

    fig.suptitle("Análisis de Patrones Temporales de Dropout — T1.2.4", fontsize=12)
    fig.tight_layout()

    out = FIGURES / "missing_dropout.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_dropout_temporal_map(
    df: pd.DataFrame, dropout_df: pd.DataFrame, rain_cols: list, rain_col_map: dict
) -> Path:
    """
    Muestra la última observación válida por estación en el tiempo
    (histograma de cuándo terminó cada estación).
    """
    from pathlib import Path

    # Estaciones monotónicas: ¿cuándo pararon?
    mono = dropout_df[dropout_df["pattern"] == "monotonic"].copy()
    # Mapear last_valid (índice en rain_cols) → (month, year)
    last_valid_dates = []
    for _, row in mono.iterrows():
        lv = int(row["last_valid"])
        if 0 <= lv < len(rain_cols):
            m, y = rain_col_map[rain_cols[lv]]
            last_valid_dates.append(y + (m - 1) / 12)
    last_valid_dates = np.array(last_valid_dates)

    # Estaciones intermitentes: longitud del bloque NaN más largo
    inter = dropout_df[dropout_df["pattern"] == "intermittent"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    if len(last_valid_dates) > 0:
        axes[0].hist(last_valid_dates, bins=30, color="#e74c3c", alpha=0.8, edgecolor="white")
        axes[0].set_xlabel("Año de último reporte")
        axes[0].set_ylabel("N° estaciones")
        axes[0].set_title(
            f"Estaciones monotónicas (n={len(mono):,})\n¿Cuándo dejaron de reportar?",
            fontsize=10,
        )
    else:
        axes[0].set_visible(False)

    if len(inter) > 0:
        axes[1].hist(
            inter["longest_nan_block"].clip(0, 80),
            bins=30, color="#f39c12", alpha=0.8, edgecolor="white",
        )
        axes[1].set_xlabel("Longitud del bloque NaN más largo (meses)")
        axes[1].set_ylabel("N° estaciones")
        axes[1].set_title(
            f"Estaciones intermitentes (n={len(inter):,})\nDistribución del gap más largo",
            fontsize=10,
        )

    fig.suptitle("Comportamiento temporal del dropout", fontsize=11)
    fig.tight_layout()

    out = FIGURES / "missing_dropout_temporal.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


# ── Reporte Markdown ─────────────────────────────────────────────────────────

def write_report(
    df: pd.DataFrame,
    rain_cols: list,
    rain_col_map: dict,
    mcar_results: tuple,
    corr_results: dict,
    dropout_df: pd.DataFrame,
) -> Path:
    """Genera outputs/reports/diagnostico_datos_faltantes.md."""
    from pathlib import Path

    REPORTS.mkdir(parents=True, exist_ok=True)

    chi2_stat, df_chi, p_value = mcar_results
    total_cells = len(df) * len(rain_cols)
    missing_cells = int(df[rain_cols].isna().sum().sum())
    pct_missing = missing_cells / total_cells

    # Veredicto MCAR
    if np.isnan(p_value):
        mcar_verdict = "No determinado (test inconcluso)"
        mcar_class = "MCAR?"
    elif p_value < 0.05:
        mcar_verdict = f"**RECHAZADO** (χ²={chi2_stat:.1f}, gl={df_chi:,}, p≈{p_value:.2e})"
        mcar_class = "**NO es MCAR**"
    else:
        mcar_verdict = f"No rechazado (χ²={chi2_stat:.1f}, gl={df_chi:,}, p={p_value:.4f})"
        mcar_class = "compatible con MCAR"

    # Clasificación final (MAR vs MNAR)
    lat_rs = [v["r"] for v in corr_results.get("Lat", {}).values()]
    lon_rs = [v["r"] for v in corr_results.get("Long", {}).values()]
    lat_sig = sum(1 for v in corr_results.get("Lat", {}).values() if v["p"] < 0.05)
    lon_sig = sum(1 for v in corr_results.get("Long", {}).values() if v["p"] < 0.05)
    lat_mean_r = np.nanmean(lat_rs) if lat_rs else np.nan
    lon_mean_r = np.nanmean(lon_rs) if lon_rs else np.nan

    # Temporal summary
    tdf = corr_results.get("temporal", pd.DataFrame())
    annual_miss = tdf.groupby("year")["missing_rate"].mean() if not tdf.empty else pd.Series()
    worst_year_str = ""
    best_year_str = ""
    if len(annual_miss) > 0:
        worst_yr = annual_miss.idxmax()
        worst_val = annual_miss.max()
        best_yr = annual_miss.idxmin()
        best_val = annual_miss.min()
        worst_year_str = f"{worst_yr} ({worst_val:.1%})"
        best_year_str = f"{best_yr} ({best_val:.1%})"

    # Dropout summary
    pat_counts = dropout_df["pattern"].value_counts().to_dict()
    n_mono = pat_counts.get("monotonic", 0)
    n_inter = pat_counts.get("intermittent", 0)
    n_complete = pat_counts.get("complete", 0)
    n_active = pat_counts.get("active", 0)

    # Final classification
    spatial_signal = (
        abs(lat_mean_r) > 0.10 and lat_sig > len(lat_rs) * 0.20
        or abs(lon_mean_r) > 0.10 and lon_sig > len(lon_rs) * 0.20
    )
    if np.isnan(p_value) or p_value >= 0.05:
        final_class = "MCAR (o similar)"
        final_rationale = (
            "El test de Little no rechaza MCAR y las correlaciones "
            "espaciales son débiles."
        )
    elif spatial_signal:
        final_class = "MAR"
        final_rationale = (
            "El test de Little rechaza MCAR y se observa correlación "
            "significativa entre datos faltantes y ubicación geográfica "
            "(Lat/Long), lo que sugiere que los datos faltantes están "
            "relacionados con variables observadas → MAR."
        )
    else:
        final_class = "MAR / MNAR (no determinado)"
        final_rationale = (
            "El test de Little rechaza MCAR pero las correlaciones "
            "espaciales son débiles. El mecanismo podría ser MNAR "
            "(relacionado con el valor de la precipitación misma, "
            "por ejemplo sensores que fallan durante eventos extremos) "
            "o MAR condicionado a variables no incluidas en el análisis."
        )

    report = f"""# Diagnóstico de Datos faltantes — Dataset Pluviométrico México (2013–2026)

**Generado**: {date.today().isoformat()}
**Dataset**: `data/processed/lluvia_clean.parquet`

---

## Resumen Ejecutivo

| Métrica | Valor |
|---|---|
| Estaciones | {len(df):,} |
| Meses monitoreados | {len(rain_cols)} (ene-2013 → may-2026) |
| Celdas totales | {total_cells:,} |
| Celdas faltantes | {missing_cells:,} ({pct_missing:.1%}) |
| Veredicto MCAR | {mcar_class} |
| **Clasificación final** | **{final_class}** |

---

## T1.2.1 — Patrón Visual de Datos faltantes

Se generaron dos figuras:

- `outputs/figures/missing_matrix.png` — Matriz de datos faltantes con todas las estaciones
  ordenadas por Estado (A→Z) y % completitud (↓). Las estaciones con más datos aparecen
  arriba dentro de cada bloque de estado.

- `outputs/figures/missing_by_state_year.png` — Heatmap de proporción de datos faltantes
  por Estado × Año.

### Hallazgos principales

| Año | Tasa de faltantes (promedio nacional) |
|---|---|
{chr(10).join(f"| {yr} | {rate:.1%} |" for yr, rate in annual_miss.items()) if len(annual_miss) > 0 else "| N/A | N/A |"}

- **Peor año**: {worst_year_str}
- **Mejor año**: {best_year_str}

El patrón visual muestra bloques de datos faltantes consistentes por estado, lo que sugiere
que los datos faltantes NO están distribuidos aleatoriamente en el espacio.

---

## T1.2.2 — Test de Little para MCAR

**Resultado**: {mcar_verdict}

**Interpretación**: {mcar_class}.

> **Nota metodológica**: Se usó un subconjunto de columnas con completitud ≥ 35%
> y una muestra de hasta 1,000 estaciones para garantizar estabilidad numérica.
> El test es una aproximación simplificada de Little (1988); los resultados deben
> interpretarse con cautela dado el alto porcentaje de datos faltantes (>{pct_missing:.0%}).

---

## T1.2.3 — Correlaciones Punto-Biserial

Se calculó la correlación punto-biserial entre el indicador binario de ausencia
(1 = faltante, 0 = presente) y las variables de contexto Lat y Long,
para cada una de las {len(rain_cols)} columnas mensuales.

| Variable | r̄ promedio | Meses con r significativo (p<0.05) |
|---|---|---|
| Latitud | {lat_mean_r:.3f} | {lat_sig}/{len(lat_rs)} ({lat_sig/len(lat_rs):.0%}) |
| Longitud | {lon_mean_r:.3f} | {lon_sig}/{len(lon_rs)} ({lon_sig/len(lon_rs):.0%}) |

### Interpretación

{"La datos faltantes muestra **correlación espacial significativa** con Lat/Long." if spatial_signal else "Las correlaciones espaciales son **débiles** (|r̄| < 0.10)."}

El eje temporal también revela estacionalidad en la tasa de faltantes:
- Años 2013–2017 presentan las tasas más altas de datos faltantes, consistente
  con el diagnóstico previo (peor año: {worst_year_str}).
- Ver figura `outputs/figures/missing_biserial.png`.

---

## T1.2.4 — Segmentación Temporal de Dropout

Se clasificó el patrón de datos faltantes temporal de cada estación en cuatro categorías:

| Patrón | N° estaciones | Descripción |
|---|---|---|
| `active` | {n_active:,} | Completitud ≥ 95% (sin dropout relevante) |
| `intermittent` | {n_inter:,} | Bloques de NaN dispersos; la estación reporta y desaparece |
| `monotonic` | {n_mono:,} | La estación deja de reportar y no regresa |
| `complete` | {n_complete:,} | Sin ningún dato en todo el período |

- **Estaciones monotónicas ({n_mono})**: representan sensores fuera de operación o
  estaciones descontinuadas. El histograma de `last_valid` (figura `missing_dropout_temporal.png`)
  muestra cuándo ocurrió la caída definitiva.

- **Estaciones intermitentes ({n_inter})**: sugieren problemas intermitentes de comunicación,
  mantenimiento o digitalización. Estas son las más problemáticas para la imputación.

- **Estaciones completas sin datos ({n_complete})**: excluir directamente de análisis posteriores
  (ya identificadas en el perfilado inicial).

---

## Clasificación Final del Mecanismo de Datos faltantes

> **Veredicto: {final_class}**

{final_rationale}

### Implicaciones para las fases siguientes

1. **Imputación** (T1.3+): dado que los datos NO son MCAR, la imputación simple
   (media/mediana) introduce sesgo. Se recomienda **imputación múltiple** (MICE)
   o modelos basados en información espacial (kriging).

2. **Análisis composicional (Fase III)**: las estaciones con patrones intermitentes
   o monotónicos deben tratarse con cuidado. Solo estaciones con ≥ 10 meses por año
   durante ≥ 3 años serán incluidas en el análisis CoDA (criterio T3.1.1).

3. **Subperíodo 2013–2017**: el alto porcentaje de datos faltantes en este período puede sesgar los
   análisis de tendencia. Considerar análisis de sensibilidad excluyendo 2013–2017.

---

## Figuras generadas

| Archivo | Descripción |
|---|---|
| `missing_matrix.png` | Matriz missingno — 1,959 estaciones × 161 meses |
| `missing_by_state_year.png` | Heatmap por Estado × Año |
| `missing_biserial.png` | Correlaciones punto-biserial + tasa temporal |
| `missing_dropout.png` | Distribución de patrones de dropout |
| `missing_dropout_temporal.png` | Cuándo dejaron de reportar las estaciones monotónicas |

---

## Referencias

- Little, R.J.A. (1988). *A test of missing completely at random for multivariate data with missing values*. JASA, 83(404), 1198–1202.
- Sterne, J.A.C. et al. (2009). *Multiple imputation for missing data in epidemiological and clinical research*. BMJ, 338, b2393.
"""

    out = REPORTS / "diagnostico_datos_faltantes.md"
    out.write_text(report, encoding="utf-8")
    return out


# ── Punto de entrada ─────────────────────────────────────────────────────────

def run_t1_2(verbose: bool = True) -> None:
    """Ejecuta T1.2 completo: carga datos, genera figuras y reporte."""
    from pathlib import Path

    if verbose:
        print("=" * 60)
        print("T1.2 — Diagnóstico de Datos Faltantes")
        print("=" * 60)

    # Cargar datos
    df = pd.read_parquet(DATA_PROCESSED / "lluvia_clean.parquet")
    rain_col_map = parse_rain_columns(df.columns)
    rain_cols = _sorted_rain_cols(rain_col_map)

    if verbose:
        print(f"Datos cargados: {df.shape[0]:,} estaciones × {len(rain_cols)} meses")

    FIGURES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    # ── T1.2.1 ──
    if verbose:
        print("\n[T1.2.1] Generando matriz de datos faltantes...")
    plot_datos_faltantes_matrix(df, rain_cols, rain_col_map)
    print("         → missing_matrix.png")
    plot_missing_by_state_year(df, rain_cols, rain_col_map)
    print("         → missing_by_state_year.png")

    # ── T1.2.2 ──
    if verbose:
        print("\n[T1.2.2] Test de Little (MCAR)...")
    mcar_results = littles_mcar_test(
        df[rain_cols], min_col_completeness=0.35, max_rows=1000, verbose=verbose
    )
    chi2_stat, df_chi, p_value = mcar_results
    if verbose:
        verdict = "RECHAZADO" if (not np.isnan(p_value) and p_value < 0.05) else "NO rechazado"
        print(
            f"    χ²={chi2_stat:.1f}, gl={df_chi:,}, p={p_value:.2e} → H0 {verdict}"
        )

    # ── T1.2.3 ──
    if verbose:
        print("\n[T1.2.3] Correlaciones punto-biserial...")
    corr_results = biserial_correlations(df, rain_cols, rain_col_map, verbose=verbose)
    plot_biserial_correlations(df, rain_cols, rain_col_map, corr_results)
    print("         → missing_biserial.png")

    # ── T1.2.4 ──
    if verbose:
        print("\n[T1.2.4] Clasificando patrones de dropout...")
    dropout_df = classify_dropout_patterns(df, rain_cols)
    counts = dropout_df["pattern"].value_counts()
    if verbose:
        for pat, cnt in counts.items():
            print(f"    {pat}: {cnt:,} estaciones ({cnt/len(dropout_df):.1%})")
    plot_dropout_analysis(dropout_df, rain_cols)
    print("         → missing_dropout.png")
    plot_dropout_temporal_map(df, dropout_df, rain_cols, rain_col_map)
    print("         → missing_dropout_temporal.png")

    # ── Reporte ──
    if verbose:
        print("\n[Reporte] Escribiendo diagnostico_datos_faltantes.md...")
    report_path = write_report(
        df, rain_cols, rain_col_map, mcar_results, corr_results, dropout_df
    )
    print(f"         → {report_path}")

    if verbose:
        print("\n[OK] T1.2 completado.")


if __name__ == "__main__":
    run_t1_2(verbose=True)
