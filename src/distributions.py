"""
Caracterización distribucional de la precipitación mensual — Tarea T1.3.

Analiza la forma, estacionalidad y estructura espacio-temporal de la
distribución de precipitación en las 1,959 estaciones del SMN. Cinco subtareas:

  T1.3.1  Histograma + KDE en escala original y log(x+1); momentos descriptivos
          globales (skewness, kurtosis, proporción de ceros).
  T1.3.2  Boxplot estacional por mes calendario y Precipitation Concentration
          Index (PCI, Oliver 1980) por estación y por estado.
  T1.3.3  Clustermap Estado × 12 meses (mediana de precipitación) con
          dendrograma jerárquico Ward/euclídea en ambos ejes.
  T1.3.4  Descomposición STL de la serie nacional (mediana mensual) y test
          de Mann-Kendall sobre el componente de tendencia.
  T1.3.5  Semivariograma empírico de la precipitación anual media (estaciones
          con pct_complete ≥ 80%) y ajuste de modelo esférico o exponencial.

Los parámetros del variograma esférico resultantes se reutilizan en T1.4 y T2.3.
Genera seis figuras y el archivo outputs/reports/distribution_summary.csv.

Punto de entrada: ``run_t1_3(verbose=True)``.

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
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.stats import skew, kurtosis, gaussian_kde
from statsmodels.tsa.seasonal import STL
import pymannkendall as mk
from pykrige.ok import OrdinaryKriging
from pathlib import Path

from src.config import DATA_PROCESSED, FIGURES, REPORTS, SEED
from src.loading import parse_rain_columns


# ── Utilidades ───────────────────────────────────────────────────────────────

def _sorted_rain_cols(rain_col_map: dict) -> list[str]:
    return sorted(rain_col_map, key=lambda c: (rain_col_map[c][1], rain_col_map[c][0]))


def _monthly_groups(rain_col_map: dict) -> dict[int, list[str]]:
    """Devuelve {mes: [cols]} para mes 1–12."""
    groups: dict[int, list[str]] = {m: [] for m in range(1, 13)}
    for col, (m, _) in rain_col_map.items():
        groups[m].append(col)
    return groups


def pci(monthly_means: np.ndarray) -> float:
    """
    Precipitation Concentration Index (Oliver, 1980).
    PCI = (Σ pᵢ²) / (Σ pᵢ)² × 100
    ≈ 8.33 → distribución uniforme
    > 20   → concentración estacional fuerte
    """
    p = np.asarray(monthly_means, dtype=float)
    p = p[~np.isnan(p)]
    if len(p) < 12 or p.sum() == 0:
        return np.nan
    return float(np.sum(p ** 2) / (np.sum(p) ** 2) * 100)


# ── T1.3.1 — Histograma + KDE ───────────────────────────────────────────────

def t1_3_1_histogram(
    df: pd.DataFrame, rain_cols: list, verbose: bool = True
) -> tuple[dict, Path]:
    """
    Histograma + KDE en escala original y log(x+1).
    Calcula skewness y kurtosis (exceso).
    """
    vals = df[rain_cols].values.flatten()
    vals = vals[~np.isnan(vals)]
    vals_log = np.log1p(vals)

    moments = {
        "n_valid": int(len(vals)),
        "mean_mm": float(np.mean(vals)),
        "median_mm": float(np.median(vals)),
        "std_mm": float(np.std(vals, ddof=1)),
        "skewness": float(skew(vals)),
        "kurtosis_excess": float(kurtosis(vals)),          # excess (Fisher)
        "pct_zeros": float((vals == 0).sum() / len(vals)),
        "max_mm": float(vals.max()),
        # log scale
        "mean_log1p": float(np.mean(vals_log)),
        "skewness_log1p": float(skew(vals_log)),
        "kurtosis_log1p": float(kurtosis(vals_log)),
    }

    if verbose:
        print(f"    n_valid={moments['n_valid']:,}  "
              f"media={moments['mean_mm']:.1f}  "
              f"mediana={moments['median_mm']:.1f}  "
              f"σ={moments['std_mm']:.1f}  "
              f"skew={moments['skewness']:.2f}  "
              f"kurt={moments['kurtosis_excess']:.2f}  "
              f"ceros={moments['pct_zeros']:.1%}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # — Panel izq: escala original (recortar en p99 para legibilidad) —
    p99 = np.percentile(vals, 99)
    vals_trunc = vals[vals <= p99]
    axes[0].hist(vals_trunc, bins=80, density=True, color="steelblue",
                 alpha=0.7, edgecolor="white", linewidth=0.3)
    kde0 = gaussian_kde(vals_trunc, bw_method="scott")
    xx0 = np.linspace(0, p99, 400)
    axes[0].plot(xx0, kde0(xx0), color="darkred", linewidth=1.8)
    axes[0].set_xlabel("Precipitación mensual (mm)")
    axes[0].set_ylabel("Densidad")
    axes[0].set_title(
        f"Escala original (recortado a P99={p99:.0f} mm)\n"
        f"Media={moments['mean_mm']:.1f}  Mediana={moments['median_mm']:.1f}  "
        f"σ={moments['std_mm']:.1f}  Skew={moments['skewness']:.2f}"
    )
    for pct, label in [(50, "P50"), (90, "P90"), (95, "P95")]:
        v = np.percentile(vals, pct)
        axes[0].axvline(v, linestyle="--", linewidth=0.9, alpha=0.7,
                        label=f"{label}={v:.0f} mm")
    axes[0].legend(fontsize=8)

    # — Panel der: log(x+1) —
    axes[1].hist(vals_log, bins=80, density=True, color="darkorange",
                 alpha=0.7, edgecolor="white", linewidth=0.3)
    kde1 = gaussian_kde(vals_log, bw_method="scott")
    xx1 = np.linspace(0, vals_log.max(), 400)
    axes[1].plot(xx1, kde1(xx1), color="darkred", linewidth=1.8)
    axes[1].set_xlabel("log(precipitación + 1)")
    axes[1].set_ylabel("Densidad")
    axes[1].set_title(
        f"Escala log(x+1)\n"
        f"Media={moments['mean_log1p']:.2f}  "
        f"Skew={moments['skewness_log1p']:.2f}  "
        f"Kurt={moments['kurtosis_log1p']:.2f}"
    )

    fig.suptitle("T1.3.1 — Distribución de la precipitación mensual (todas las estaciones)",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    out = FIGURES / "dist_histogram.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return moments, out


# ── T1.3.2 — Boxplot estacional + PCI ───────────────────────────────────────

def t1_3_2_seasonal(
    df: pd.DataFrame, rain_cols: list, rain_col_map: dict, verbose: bool = True
) -> tuple[pd.DataFrame, list[Path]]:
    """
    Boxplot estacional por mes (1–12) y PCI por estación.
    """
    mg = _monthly_groups(rain_col_map)
    MONTH_NAMES = ["Ene","Feb","Mar","Abr","May","Jun",
                   "Jul","Ago","Sep","Oct","Nov","Dic"]

    # Recopilar valores por mes para el boxplot
    month_data: list[np.ndarray] = []
    monthly_stats: list[dict] = []
    for m in range(1, 13):
        vals_m = df[mg[m]].values.flatten()
        vals_m = vals_m[~np.isnan(vals_m)]
        month_data.append(vals_m)
        monthly_stats.append({
            "month": m, "name": MONTH_NAMES[m - 1],
            "median_mm": float(np.median(vals_m)) if len(vals_m) else np.nan,
            "mean_mm": float(np.mean(vals_m)) if len(vals_m) else np.nan,
            "p90_mm": float(np.percentile(vals_m, 90)) if len(vals_m) else np.nan,
            "n_valid": int(len(vals_m)),
        })

    if verbose:
        print("    Medianas mensuales (mm):", " ".join(
            f"{s['name']}:{s['median_mm']:.0f}" for s in monthly_stats))

    # Figura 1: Boxplot estacional
    fig1, ax1 = plt.subplots(figsize=(14, 6))
    bp = ax1.boxplot(
        month_data, tick_labels=MONTH_NAMES,
        showfliers=False, patch_artist=True,
        medianprops=dict(color="black", linewidth=1.5),
    )
    colors_season = (
        ["#aec6e8"] * 4 +          # Ene–Abr (seco)
        ["#2196F3"] * 6 +          # May–Oct (húmedo)
        ["#aec6e8"] * 2            # Nov–Dic (seco)
    )
    for patch, color in zip(bp["boxes"], colors_season):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    ax1.set_xlabel("Mes")
    ax1.set_ylabel("Precipitación mensual (mm)")
    ax1.set_title("T1.3.2 — Boxplot estacional (sin valores atípicos)\n"
                  "Azul oscuro = meses húmedos (mayo–octubre)")
    ax1.grid(axis="y", alpha=0.3)

    # Mediana sobre cada caja
    for i, s in enumerate(monthly_stats):
        ax1.text(i + 1, s["median_mm"] + 2, f"{s['median_mm']:.0f}",
                 ha="center", va="bottom", fontsize=7.5)

    out1 = FIGURES / "dist_seasonal_boxplot.png"
    fig1.tight_layout()
    fig1.savefig(out1, dpi=140, bbox_inches="tight")
    plt.close(fig1)

    # PCI por estación
    monthly_means_mat = pd.DataFrame(
        {m: df[mg[m]].mean(axis=1) for m in range(1, 13)}
    )
    pci_vals = monthly_means_mat.apply(lambda row: pci(row.values), axis=1)
    df_pci = pd.DataFrame({
        "#Station": df["#Station"],
        "State": df["State"],
        "pci": pci_vals,
        "pct_complete": df["pct_complete"],
    })
    df_pci_valid = df_pci.dropna(subset=["pci"])

    if verbose:
        print(f"    PCI — n={len(df_pci_valid):,}  "
              f"media={df_pci_valid['pci'].mean():.2f}  "
              f"mediana={df_pci_valid['pci'].median():.2f}  "
              f"max={df_pci_valid['pci'].max():.2f}")

    # Figura 2: distribución del PCI por estado
    pci_by_state = (
        df_pci_valid.groupby("State")["pci"]
        .median()
        .sort_values(ascending=False)
    )

    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))

    # Panel izq: histograma del PCI nacional
    axes2[0].hist(df_pci_valid["pci"], bins=40, color="teal", alpha=0.8,
                  edgecolor="white", linewidth=0.3)
    axes2[0].axvline(8.33, color="green", linestyle="--", linewidth=1.2,
                     label="Uniforme (8.33)")
    axes2[0].axvline(20, color="orange", linestyle="--", linewidth=1.2,
                     label="Umbral alta conc. (20)")
    axes2[0].set_xlabel("PCI")
    axes2[0].set_ylabel("N° estaciones")
    axes2[0].set_title(
        f"Distribución nacional del PCI (n={len(df_pci_valid):,})\n"
        f"Mediana={df_pci_valid['pci'].median():.1f}  "
        f"P90={df_pci_valid['pci'].quantile(0.9):.1f}"
    )
    axes2[0].legend(fontsize=8)

    # Panel der: PCI mediano por estado
    axes2[1].barh(pci_by_state.index, pci_by_state.values,
                  color="teal", alpha=0.8, edgecolor="white")
    axes2[1].axvline(8.33, color="green", linestyle="--", linewidth=1)
    axes2[1].axvline(20, color="orange", linestyle="--", linewidth=1)
    axes2[1].set_xlabel("PCI mediano")
    axes2[1].set_title("PCI mediano por Estado\n(↑ mayor concentración estacional)")

    fig2.suptitle("T1.3.2 — Precipitation Concentration Index (Oliver, 1980)", fontsize=11)
    fig2.tight_layout()
    out2 = FIGURES / "dist_pci.png"
    fig2.savefig(out2, dpi=140, bbox_inches="tight")
    plt.close(fig2)

    monthly_stats_df = pd.DataFrame(monthly_stats)
    return monthly_stats_df, df_pci_valid, [out1, out2]


# ── T1.3.3 — Heatmap Estado × 12 meses ─────────────────────────────────────

def t1_3_3_state_month_heatmap(
    df: pd.DataFrame, rain_cols: list, rain_col_map: dict, verbose: bool = True
) -> Path:
    """
    Heatmap 32 estados × 12 meses con mediana de precipitación.
    Clustermap con dendrograma en ambos ejes.
    """
    mg = _monthly_groups(rain_col_map)
    MONTH_NAMES = ["Ene","Feb","Mar","Abr","May","Jun",
                   "Jul","Ago","Sep","Oct","Nov","Dic"]
    states = sorted(df["State"].unique())

    # Matriz 32 × 12: mediana de precipitación mensual por estado
    matrix = pd.DataFrame(index=states, columns=list(range(1, 13)), dtype=float)
    for m in range(1, 13):
        for state in states:
            vals = df.loc[df["State"] == state, mg[m]].values.flatten()
            vals = vals[~np.isnan(vals)]
            matrix.loc[state, m] = float(np.median(vals)) if len(vals) else np.nan
    matrix.columns = MONTH_NAMES

    if verbose:
        print(f"    Matriz Estado×Mes generada: {matrix.shape} "
              f"(NaN: {matrix.isna().sum().sum()} celdas)")

    # Rellenar NaN con la mediana de la columna para que el clustering no falle
    matrix_fill = matrix.fillna(matrix.median())

    g = sns.clustermap(
        matrix_fill,
        figsize=(14, 13),
        cmap="YlOrRd",
        linewidths=0.3,
        annot=matrix.round(0),     # mostrar valores originales (con NaN como "")
        annot_kws={"size": 7},
        fmt=".0f",
        method="ward",
        metric="euclidean",
        row_cluster=True,
        col_cluster=True,
        cbar_kws={"label": "Mediana precipitación (mm)", "shrink": 0.6},
        yticklabels=True,
        xticklabels=True,
    )
    g.ax_heatmap.set_xlabel("Mes", fontsize=10)
    g.ax_heatmap.set_ylabel("Estado", fontsize=10)
    g.fig.suptitle(
        "T1.3.3 — Mediana de precipitación mensual por Estado × Mes\n"
        "(clustermap Ward / euclídea)",
        fontsize=11, y=1.01,
    )

    out = FIGURES / "dist_state_month_clustermap.png"
    g.fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(g.fig)
    return matrix, out


# ── T1.3.4 — STL + Mann-Kendall ─────────────────────────────────────────────

def t1_3_4_stl_mannkendall(
    df: pd.DataFrame, rain_cols: list, rain_col_map: dict, verbose: bool = True
) -> tuple[dict, Path]:
    """
    Descomposición STL de la serie nacional (mediana mensual).
    Test de Mann-Kendall sobre el componente de tendencia.
    """
    # Serie temporal: mediana nacional mensual
    national_median = df[rain_cols].median(axis=0)
    dates = pd.date_range("2013-01", periods=len(rain_cols), freq="MS")
    ts = pd.Series(national_median.values, index=dates, name="precip_mm")

    if verbose:
        print(f"    Serie nacional: n={len(ts)}, "
              f"media={ts.mean():.1f} mm, "
              f"min={ts.min():.1f}, max={ts.max():.1f}")

    # STL
    stl = STL(ts, period=12, robust=True)
    result = stl.fit()

    # Mann-Kendall sobre la tendencia STL; si falla, sobre la serie completa.
    # Se aplica primero sobre trend para eliminar la componente estacional,
    # que infla el estadístico si se usa la serie cruda.
    trend_series = result.trend
    for _src, _vals in [
        ("trend STL", trend_series.dropna().values.astype(float)),
        ("serie raw",  ts.dropna().values.astype(float)),
    ]:
        try:
            mk_result = mk.original_test(_vals)
            mk_source = _src
            break
        except (ZeroDivisionError, ValueError, Exception):
            continue
    else:
        # Fallback manual: kendall τ con scipy
        from scipy.stats import kendalltau, theilslopes
        x_idx = np.arange(len(ts))
        tau, p_val = kendalltau(x_idx, ts.values)
        slope, _, _, _ = theilslopes(ts.values, x_idx)
        class _MKResult:
            trend = "increasing" if tau > 0 else ("decreasing" if tau < 0 else "no trend")
            h = p_val < 0.05
            p = p_val
            z = float(tau)
            Tau = float(tau)
            slope = float(slope)
            intercept = float(ts.values[0])
        mk_result = _MKResult()
        mk_source = "kendall τ (fallback)"

    if verbose:
        print(f"    Mann-Kendall ({mk_source}) — tendencia: '{mk_result.trend}', "
              f"slope={mk_result.slope:.4f} mm/mes, "
              f"p={mk_result.p:.4f}")

    stl_stats = {
        "trend_verdict": mk_result.trend,
        "mk_slope_mm_per_month": float(mk_result.slope),
        "mk_p_value": float(mk_result.p),
        "mk_tau": float(mk_result.Tau),
        "seasonal_amplitude_mm": float(result.seasonal.max() - result.seasonal.min()),
        "residual_std_mm": float(result.resid.std()),
    }

    # Figura: 4 paneles (observado, tendencia, estacionalidad, residuo)
    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)

    axes[0].plot(ts.index, ts.values, color="steelblue", linewidth=1)
    axes[0].set_ylabel("mm")
    axes[0].set_title("Observado (mediana nacional mensual)")

    axes[1].plot(ts.index, result.trend, color="darkred", linewidth=1.5)
    slope_str = f"{mk_result.slope:+.3f} mm/mes"
    p_str = f"p={mk_result.p:.3f}"
    axes[1].set_ylabel("mm")
    axes[1].set_title(
        f"Tendencia STL — Mann-Kendall: '{mk_result.trend}', "
        f"slope={slope_str}, {p_str}"
    )

    axes[2].fill_between(ts.index, result.seasonal, alpha=0.7, color="darkorange")
    axes[2].set_ylabel("mm")
    axes[2].set_title(f"Componente estacional (amplitud={stl_stats['seasonal_amplitude_mm']:.1f} mm)")

    axes[3].scatter(ts.index, result.resid, s=6, alpha=0.7, color="gray")
    axes[3].axhline(0, color="black", linewidth=0.6)
    axes[3].set_ylabel("mm")
    axes[3].set_title(f"Residuo (σ={stl_stats['residual_std_mm']:.1f} mm)")
    axes[3].set_xlabel("Fecha")

    fig.suptitle("T1.3.4 — Descomposición STL de la serie nacional de precipitación",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    out = FIGURES / "dist_stl.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return stl_stats, out


# ── T1.3.5 — Semivariograma empírico ────────────────────────────────────────

def t1_3_5_variogram(
    df: pd.DataFrame, rain_cols: list, verbose: bool = True
) -> tuple[dict, Path]:
    """
    Semivariograma empírico de la precipitación anual media.
    Restringe a estaciones con pct_complete ≥ 0.80.
    Ajusta modelo esférico (fallback: exponencial).
    """
    mask_complete = df["pct_complete"] >= 0.80
    if verbose:
        print(f"    Estaciones con pct_complete ≥ 80%: {mask_complete.sum()}")

    annual_mean = df[rain_cols].mean(axis=1)
    lons = df.loc[mask_complete, "Long"].values
    lats = df.loc[mask_complete, "Lat"].values
    zvals = annual_mean[mask_complete].values

    # Descartar NaN residuales en coordenadas
    valid_xy = ~(np.isnan(lons) | np.isnan(lats) | np.isnan(zvals))
    lons, lats, zvals = lons[valid_xy], lats[valid_xy], zvals[valid_xy]
    if verbose:
        print(f"    Estaciones válidas para variograma: {len(lons)}")

    variogram_stats = {}

    for model in ("spherical", "exponential"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                OK = OrdinaryKriging(
                    lons, lats, zvals,
                    variogram_model=model,
                    verbose=False,
                    enable_plotting=False,
                    nlags=20,
                )
            variogram_stats["model"] = model
            variogram_stats["params"] = OK.variogram_model_parameters
            variogram_stats["lags"] = OK.lags
            variogram_stats["semivariance"] = OK.semivariance
            if verbose:
                print(f"    Modelo {model}: parámetros={OK.variogram_model_parameters}")
            break
        except Exception as e:
            if verbose:
                print(f"    Modelo {model} falló: {e}. Intentando siguiente...")
            continue

    if "lags" not in variogram_stats:
        if verbose:
            print("    [aviso] No se pudo ajustar ningún variograma.")
        return {}, FIGURES / "dist_variogram.png"

    lags = variogram_stats["lags"]
    semivar = variogram_stats["semivariance"]
    params = variogram_stats["params"]
    model_name = variogram_stats["model"]

    # Curva del modelo ajustado
    lags_fine = np.linspace(0, lags.max(), 200)
    if model_name == "spherical":
        def spherical(h, psill, r, nugget):
            h = np.asarray(h)
            gamma = nugget + psill * (1.5 * (h / r) - 0.5 * (h / r) ** 3)
            gamma[h > r] = nugget + psill
            gamma[h == 0] = 0
            return gamma
        model_curve = spherical(lags_fine, *params)
        param_str = (f"psill={params[0]:.1f}  range={params[1]:.3f}°  "
                     f"nugget={params[2]:.1f}")
    else:
        def exponential(h, psill, r, nugget):
            h = np.asarray(h)
            return nugget + psill * (1 - np.exp(-h / r))
        model_curve = exponential(lags_fine, *params)
        param_str = (f"psill={params[0]:.1f}  range={params[1]:.3f}°  "
                     f"nugget={params[2]:.1f}")

    # Figura
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(lags, semivar, color="steelblue", s=50, zorder=3,
               label="Semivariograma empírico")
    ax.plot(lags_fine, model_curve, color="darkred", linewidth=2,
            label=f"Modelo {model_name}\n{param_str}")
    ax.set_xlabel("Distancia (grados lon/lat)")
    ax.set_ylabel("Semivarianza (mm²)")
    ax.set_title(
        f"T1.3.5 — Semivariograma empírico (precipitación anual media)\n"
        f"n={len(lons)} estaciones con pct_complete ≥ 80%"
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    out = FIGURES / "dist_variogram.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)

    return {
        "model": model_name,
        "nugget": float(params[2]),
        "psill": float(params[0]),
        "range_deg": float(params[1]),
    }, out


# ── Tabla resumen ────────────────────────────────────────────────────────────

def write_summary_csv(
    moments: dict,
    monthly_stats_df: pd.DataFrame,
    pci_df: pd.DataFrame,
    stl_stats: dict,
    variogram_stats: dict,
) -> Path:
    """Genera outputs/reports/distribution_summary.csv (formato largo)."""
    REPORTS.mkdir(parents=True, exist_ok=True)

    rows = []

    # Momentos globales
    for k, v in moments.items():
        rows.append({"section": "T1.3.1_global", "metric": k, "value": v})

    # Estadísticos mensuales
    for _, row in monthly_stats_df.iterrows():
        m = row["name"]
        for col in ["median_mm", "mean_mm", "p90_mm", "n_valid"]:
            rows.append({"section": "T1.3.2_monthly", "metric": f"{m}_{col}", "value": row[col]})

    # PCI nacional
    rows.append({"section": "T1.3.2_pci", "metric": "pci_national_median",
                 "value": float(pci_df["pci"].median())})
    rows.append({"section": "T1.3.2_pci", "metric": "pci_national_mean",
                 "value": float(pci_df["pci"].mean())})
    rows.append({"section": "T1.3.2_pci", "metric": "pci_national_p90",
                 "value": float(pci_df["pci"].quantile(0.9))})

    # STL + Mann-Kendall
    for k, v in stl_stats.items():
        rows.append({"section": "T1.3.4_stl", "metric": k, "value": v})

    # Variograma
    for k, v in variogram_stats.items():
        if k not in ("lags", "semivariance", "params"):
            rows.append({"section": "T1.3.5_variogram", "metric": k, "value": v})

    df_out = pd.DataFrame(rows)
    out = REPORTS / "distribution_summary.csv"
    df_out.to_csv(out, index=False)
    return out


# ── Punto de entrada ─────────────────────────────────────────────────────────

def run_t1_3(verbose: bool = True) -> None:
    """Ejecuta T1.3 completo."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("=" * 60)
        print("T1.3 — Caracterización Distribucional")
        print("=" * 60)

    df = pd.read_parquet(DATA_PROCESSED / "lluvia_clean.parquet")
    rain_col_map = parse_rain_columns(df.columns)
    rain_cols = _sorted_rain_cols(rain_col_map)

    if verbose:
        print(f"Datos: {df.shape[0]:,} estaciones × {len(rain_cols)} meses\n")

    # T1.3.1
    if verbose:
        print("[T1.3.1] Histograma + KDE...")
    moments, _ = t1_3_1_histogram(df, rain_cols, verbose=verbose)
    print("         → dist_histogram.png")

    # T1.3.2
    if verbose:
        print("\n[T1.3.2] Boxplot estacional + PCI...")
    monthly_stats_df, pci_df, _ = t1_3_2_seasonal(
        df, rain_cols, rain_col_map, verbose=verbose
    )
    print("         → dist_seasonal_boxplot.png")
    print("         → dist_pci.png")

    # T1.3.3
    if verbose:
        print("\n[T1.3.3] Heatmap Estado × Mes (clustermap)...")
    state_month_matrix, _ = t1_3_3_state_month_heatmap(
        df, rain_cols, rain_col_map, verbose=verbose
    )
    print("         → dist_state_month_clustermap.png")

    # T1.3.4
    if verbose:
        print("\n[T1.3.4] STL + Mann-Kendall...")
    stl_stats, _ = t1_3_4_stl_mannkendall(
        df, rain_cols, rain_col_map, verbose=verbose
    )
    print("         → dist_stl.png")

    # T1.3.5
    if verbose:
        print("\n[T1.3.5] Semivariograma empírico...")
    variogram_stats, _ = t1_3_5_variogram(df, rain_cols, verbose=verbose)
    print("         → dist_variogram.png")

    # Tabla resumen
    if verbose:
        print("\n[Resumen] Escribiendo distribution_summary.csv...")
    csv_path = write_summary_csv(moments, monthly_stats_df, pci_df, stl_stats, variogram_stats)
    print(f"         → {csv_path}")

    if verbose:
        print("\n[OK] T1.3 completado.")
        print(f"\nResumen ejecutivo:")
        print(f"  Media nacional:  {moments['mean_mm']:.1f} mm/mes")
        print(f"  Mediana nacional:{moments['median_mm']:.1f} mm/mes")
        print(f"  Skewness:        {moments['skewness']:.2f}")
        print(f"  PCI mediano:     {pci_df['pci'].median():.1f}")
        print(f"  Tendencia MK:    {stl_stats.get('trend_verdict','?')} "
              f"(slope={stl_stats.get('mk_slope_mm_per_month',0):+.4f} mm/mes, "
              f"p={stl_stats.get('mk_p_value',1):.3f})")
        if variogram_stats:
            print(f"  Variograma:      modelo={variogram_stats.get('model','?')} "
                  f"range={variogram_stats.get('range_deg',0):.2f}°")


if __name__ == "__main__":
    run_t1_3(verbose=True)
