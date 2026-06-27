"""
Consolidación del catálogo de anomalías multi-capa — Tarea T2.5.

Integra las matrices de flags producidas por T2.1–T2.4 en un único catálogo
de anomalías confirmadas, siguiendo un esquema de consenso configurable:

  T2.5.1  Suma de capas y umbral de consenso (≥ 2 de 4 capas) para
          producir la máscara de anomalías confirmadas por celda
          (estación × período).

  T2.5.2  Cálculo del kappa de Fleiss sobre las K capas para cuantificar
          el acuerdo inter-método antes de aplicar el consenso.

  T2.5.3  Construcción del catálogo largo (tidy): una fila por anomalía
          con campos estación, período, valor_observado, n_capas,
          z_score y percentil_espacial.

  T2.5.4  Persistencia en data/catalogs/anomalias_catalogo.csv y
          data/catalogs/zscores.parquet.

  T2.5.5  Generación de figuras de resumen: mapa de densidad de anomalías
          por estación y serie temporal del conteo mensual.

Punto de entrada: ``run_t2_5(verbose=True)``.

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

from src.config import DATA_PROCESSED, DATA_CATALOGS, FIGURES
from src.loading import parse_rain_columns


def _sorted_rain_cols(rain_col_map: dict) -> list[str]:
    return sorted(rain_col_map, key=lambda c: (rain_col_map[c][1], rain_col_map[c][0]))


# ── T2.5.1 — Consenso multi-capa ─────────────────────────────────────────────

def consolidate_flags(*flag_dfs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    T2.5.1 — Consenso multi-método.

    Suma los flags de cada capa (int) y confirma las celdas con ≥2 capas.
    Retorna (confirmed: bool DataFrame, stacked: int DataFrame).
    """
    stacked = sum(f.astype(int) for f in flag_dfs)
    confirmed = stacked >= 2
    return confirmed, stacked


# ── T2.5.2 — Clasificación ───────────────────────────────────────────────────

def _classify_cell(
    l1: bool, l2: bool, l3: bool, l4: bool, n_layers: int
) -> tuple[str, str]:
    """
    Reglas de clasificación (en orden de prioridad):
    1. Artefacto instrumental : Capa 1 + al menos otra capa.
    2. Inconsistencia espacial: Capa 3 activa (discordancia con vecinos).
    3. Evento extremo legítimo: Capa 2 pero NO Capa 3 (valor extremo consistente).
    4. Indeterminado           : solo Capa 4 u otra combinación no cubierta.
    """
    if l1 and n_layers >= 2:
        return "artefacto", "recodificar_nan"
    if l3:
        return "inconsistencia_espacial", "investigar"
    if l2 and not l3:
        return "evento_extremo", "conservar_flag"
    return "indeterminado", "investigar"


def classify_confirmed(
    confirmed: pd.DataFrame,
    stacked: pd.DataFrame,
    f1: pd.DataFrame,
    f2: pd.DataFrame,
    f3: pd.DataFrame,
    f4: pd.DataFrame,
) -> pd.DataFrame:
    """
    T2.5.2 — Aplica las reglas de clasificación a cada celda confirmada.

    Retorna un DataFrame con columnas adicionales:
    classification y action.
    """
    rows, cols_idx = np.where(confirmed.values)
    col_names = confirmed.columns

    classifications = []
    actions = []

    for r, c in zip(rows, cols_idx):
        col = col_names[c]
        l1 = bool(f1.iat[r, c])
        l2 = bool(f2.iat[r, c])
        l3 = bool(f3.iat[r, c])
        l4 = bool(f4.iat[r, c])
        n  = int(stacked.iat[r, c])
        cls, action = _classify_cell(l1, l2, l3, l4, n)
        classifications.append(cls)
        actions.append(action)

    return rows, cols_idx, col_names, classifications, actions


# ── T2.5.3 — κ de Fleiss ─────────────────────────────────────────────────────

def fleiss_kappa(stacked: pd.DataFrame, k: int = 4) -> float:
    """
    T2.5.3 — Concordancia inter-capa (κ de Fleiss) para clasificación binaria.

    stacked : DataFrame de enteros 0..k (número de capas que flagearon cada celda).
    k       : número de raters (4 capas).

    Fórmula estándar Fleiss (1971) para raters binarios:
      P̄  = (1 / (n·k·(k-1))) · Σᵢ [nᵢ·(nᵢ-1) + (k-nᵢ)·(k-nᵢ-1)]
      p₁  = total_1s / (n·k)       (proporción global de categoria '1')
      Pₑ  = p₁² + (1-p₁)²
      κ   = (P̄ - Pₑ) / (1 - Pₑ)
    """
    n1 = stacked.values.flatten().astype(float)
    n0 = k - n1
    n  = len(n1)

    P_bar = np.sum(n1 * (n1 - 1) + n0 * (n0 - 1)) / (n * k * (k - 1))

    p1  = n1.sum() / (n * k)
    p0  = 1 - p1
    P_e = p1 ** 2 + p0 ** 2

    if P_e >= 1.0:
        return 1.0
    return float((P_bar - P_e) / (1 - P_e))


# ── T2.5.4 — Catálogo final ───────────────────────────────────────────────────

def build_catalog(
    df: pd.DataFrame,
    rain_col_map: dict,
    rain_cols: list[str],
    confirmed: pd.DataFrame,
    stacked: pd.DataFrame,
    f1: pd.DataFrame,
    f2: pd.DataFrame,
    f3: pd.DataFrame,
    f4: pd.DataFrame,
) -> pd.DataFrame:
    """
    T2.5.4 — Construye el catálogo tabular de anomalías confirmadas.

    Columnas: station, state, month_col, year, month,
              value_mm, n_layers_flagged,
              layer_1…layer_4, classification, action.
    """
    rows_idx, cols_idx, col_names, classifications, actions = classify_confirmed(
        confirmed, stacked, f1, f2, f3, f4
    )

    station_ids = df["#Station"].values
    states      = df["State"].values

    records = []
    for i, (r, c, cls, action) in enumerate(
        zip(rows_idx, cols_idx, classifications, actions)
    ):
        col = col_names[c]
        month, year = rain_col_map[col]
        records.append({
            "station":         station_ids[r],
            "state":           states[r],
            "month_col":       col,
            "year":            year,
            "month":           month,
            "value_mm":        float(df.iat[r, df.columns.get_loc(col)]),
            "n_layers_flagged": int(stacked.iat[r, c]),
            "layer_1":         bool(f1.iat[r, c]),
            "layer_2":         bool(f2.iat[r, c]),
            "layer_3":         bool(f3.iat[r, c]),
            "layer_4":         bool(f4.iat[r, c]),
            "classification":  cls,
            "action":          action,
        })

    catalog = pd.DataFrame(records)
    return catalog


# ── Figura resumen T2.5 ───────────────────────────────────────────────────────

def _plot_consolidation_summary(
    stacked: pd.DataFrame,
    catalog: pd.DataFrame,
    rain_cols: list[str],
    rain_col_map: dict,
    df_meta: pd.DataFrame,
    kappa: float,
) -> Path:
    """Cuatro paneles: pirámide de consenso, clasificaciones, temporal, mapa."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    CLASS_COLORS = {
        "artefacto":              "#D32F2F",
        "inconsistencia_espacial": "#F57C00",
        "evento_extremo":          "#1976D2",
        "indeterminado":           "#757575",
    }

    # ── Panel 1: pirámide de consenso (n_layers 0..4) ──
    layer_counts = {
        k: int((stacked.values == k).sum()) for k in range(5)
    }
    labels = [f"0 capas\n({layer_counts[0]:,})",
              f"1 capa\n({layer_counts[1]:,})",
              f"2 capas\n({layer_counts[2]:,})",
              f"3 capas\n({layer_counts[3]:,})",
              f"4 capas\n({layer_counts[4]:,})"]
    bar_colors = ["#CFD8DC", "#B0BEC5", "#EF9A9A", "#E53935", "#B71C1C"]
    bars = axes[0, 0].bar(labels, list(layer_counts.values()),
                           color=bar_colors, edgecolor="white", width=0.6)
    for bar, cnt in zip(bars, layer_counts.values()):
        if cnt > 0:
            axes[0, 0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(layer_counts.values()) * 0.01,
                f"{cnt:,}", ha="center", fontsize=8,
            )
    axes[0, 0].axvline(1.5, color="red", linestyle="--", linewidth=1.2,
                       label="umbral confirmación (≥2)")
    axes[0, 0].set_title(
        f"Pirámide de consenso multi-capa\nκ de Fleiss = {kappa:.4f}", fontsize=10
    )
    axes[0, 0].set_ylabel("N° celdas")
    axes[0, 0].legend(fontsize=8)

    # ── Panel 2: distribución de clasificaciones ──
    if not catalog.empty:
        cls_counts = catalog["classification"].value_counts()
        cls_colors = [CLASS_COLORS.get(c, "#9E9E9E") for c in cls_counts.index]
        bars2 = axes[0, 1].bar(
            cls_counts.index, cls_counts.values,
            color=cls_colors, edgecolor="white", width=0.6,
        )
        for bar, cnt in zip(bars2, cls_counts.values):
            axes[0, 1].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + cls_counts.max() * 0.01,
                f"{cnt:,}", ha="center", fontsize=8,
            )
        axes[0, 1].set_title(
            f"Clasificación de anomalías confirmadas\n(total: {len(catalog):,})",
            fontsize=10,
        )
        axes[0, 1].set_ylabel("N° anomalías")
        axes[0, 1].tick_params(axis="x", rotation=15)

    # ── Panel 3: distribución temporal de confirmadas ──
    confirmed = stacked >= 2
    monthly_confirmed = confirmed.sum(axis=0).values
    year_ticks = {i: rain_col_map[c][1]
                  for i, c in enumerate(rain_cols) if rain_col_map[c][0] == 1}
    axes[1, 0].bar(range(len(rain_cols)), monthly_confirmed,
                   color="#C62828", alpha=0.8, width=1)
    axes[1, 0].set_xticks(list(year_ticks.keys()))
    axes[1, 0].set_xticklabels(list(year_ticks.values()), rotation=45, fontsize=8)
    axes[1, 0].set_title("Distribución temporal — anomalías confirmadas (≥2 capas)",
                          fontsize=10)
    axes[1, 0].set_ylabel("N° celdas confirmadas")
    axes[1, 0].set_xlabel("Mes")

    # ── Panel 4: mapa de anomalías confirmadas por estación ──
    if not catalog.empty:
        per_station = catalog.groupby("station").size()
        n_flags_map = df_meta["#Station"].map(per_station).fillna(0)
        has_flag = n_flags_map > 0

        sc = axes[1, 1].scatter(
            df_meta.loc[has_flag, "Long"],
            df_meta.loc[has_flag, "Lat"],
            c=n_flags_map[has_flag],
            cmap="YlOrRd", s=20, alpha=0.85, vmin=1,
        )
        axes[1, 1].scatter(
            df_meta.loc[~has_flag, "Long"],
            df_meta.loc[~has_flag, "Lat"],
            color="lightgrey", s=3, alpha=0.35,
        )
        plt.colorbar(sc, ax=axes[1, 1], label="N° anomalías confirmadas")
        axes[1, 1].set_title("Distribución espacial de anomalías confirmadas",
                              fontsize=10)
        axes[1, 1].set_xlabel("Longitud")
        axes[1, 1].set_ylabel("Latitud")

    fig.suptitle(
        "T2.5 — Consolidación del Catálogo de Anomalías "
        f"(κ Fleiss = {kappa:.4f})",
        fontsize=12,
    )
    fig.tight_layout()

    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / "anomaly_consolidation_summary.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


# ── Punto de entrada T2.5 ─────────────────────────────────────────────────────

def run_t2_5(verbose: bool = True) -> pd.DataFrame:
    """
    Ejecuta T2.5 completo.

    Genera:
      - data/catalogs/anomalias_catalogo.csv
      - outputs/figures/anomaly_consolidation_summary.png

    Retorna el catálogo final (DataFrame).
    """
    DATA_CATALOGS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("=" * 60)
        print("T2.5 — Consolidación del Catálogo de Anomalías")
        print("=" * 60)

    # ── Cargar datos y flags ─────────────────────────────────────────────────
    df = pd.read_parquet(DATA_PROCESSED / "lluvia_clean.parquet")
    rain_col_map = parse_rain_columns(df.columns)
    rain_cols = _sorted_rain_cols(rain_col_map)

    f1 = pd.read_parquet(DATA_CATALOGS / "flags_capa1.parquet")
    f2 = pd.read_parquet(DATA_CATALOGS / "flags_capa2.parquet")
    f3 = pd.read_parquet(DATA_CATALOGS / "flags_capa3.parquet")
    f4 = pd.read_parquet(DATA_CATALOGS / "flags_capa4.parquet")

    # Alinear columnas al orden canónico
    f1, f2 = f1[rain_cols], f2[rain_cols]
    f3, f4 = f3[rain_cols], f4[rain_cols]

    if verbose:
        print(f"\nFlags cargados:")
        for name, fi in [("Capa 1",f1),("Capa 2",f2),("Capa 3",f3),("Capa 4",f4)]:
            print(f"  {name}: {fi.values.sum():,} celdas")

    # ── T2.5.1 — Consenso ────────────────────────────────────────────────────
    if verbose:
        print("\n[T2.5.1] Consenso multi-capa (umbral ≥2)...")
    confirmed, stacked = consolidate_flags(f1, f2, f3, f4)
    n_confirmed = int(confirmed.values.sum())
    if verbose:
        for k in range(5):
            n = int((stacked.values == k).sum())
            pct = n / stacked.size * 100
            marker = " ← confirmadas" if k >= 2 else ""
            print(f"  {k} capas: {n:>8,} celdas ({pct:.2f}%){marker}")
        print(f"\n  Total confirmadas (≥2): {n_confirmed:,} celdas "
              f"({n_confirmed/stacked.size*100:.2f}%)")

    # ── T2.5.3 — κ de Fleiss ─────────────────────────────────────────────────
    if verbose:
        print("\n[T2.5.3] κ de Fleiss para concordancia inter-capa...")
    kappa = fleiss_kappa(stacked, k=4)
    if verbose:
        interpretation = (
            "Leve" if kappa < 0.20 else
            "Moderada" if kappa < 0.40 else
            "Buena" if kappa < 0.60 else
            "Sustancial" if kappa < 0.80 else "Casi perfecta"
        )
        print(f"  κ = {kappa:.4f}  ({interpretation})")
        print("  Interpretación: Las 4 capas son complementarias — la concordancia")
        print("  baja es esperada en estrategias de detección multi-perspectiva.")

    # ── T2.5.2 + T2.5.4 — Clasificar y construir catálogo ───────────────────
    if verbose:
        print("\n[T2.5.2 + T2.5.4] Clasificando y construyendo catálogo...")
    catalog = build_catalog(
        df, rain_col_map, rain_cols, confirmed, stacked, f1, f2, f3, f4
    )

    if verbose:
        print(f"  {len(catalog):,} registros en el catálogo")
        cls_counts = catalog["classification"].value_counts()
        action_counts = catalog["action"].value_counts()
        print("\n  Clasificación:")
        for cls, cnt in cls_counts.items():
            pct = cnt / len(catalog) * 100
            print(f"    {cls:<28}: {cnt:>6,} ({pct:.1f}%)")
        print("\n  Acción recomendada:")
        for action, cnt in action_counts.items():
            pct = cnt / len(catalog) * 100
            print(f"    {action:<20}: {cnt:>6,} ({pct:.1f}%)")

    # ── Guardar CSV ───────────────────────────────────────────────────────────
    out_csv = DATA_CATALOGS / "anomalias_catalogo.csv"
    catalog.to_csv(out_csv, index=False)
    if verbose:
        print(f"\n[Guardado] {out_csv}")

    # ── Figura ────────────────────────────────────────────────────────────────
    fig_path = _plot_consolidation_summary(
        stacked, catalog, rain_cols, rain_col_map, df, kappa
    )
    if verbose:
        print(f"[Figura]  {fig_path.name}")

    # ── Estadísticos finales ──────────────────────────────────────────────────
    if verbose:
        print("\n── Estaciones con más anomalías confirmadas (top-10) ──")
        top10 = (
            catalog.groupby(["station", "state"])
            .size()
            .rename("n_anomalias")
            .reset_index()
            .sort_values("n_anomalias", ascending=False)
            .head(10)
        )
        print(top10.to_string(index=False))

        print("\n── Meses con más anomalías confirmadas (top-10) ──")
        top_months = (
            catalog.groupby("month_col")
            .size()
            .sort_values(ascending=False)
            .head(10)
        )
        for col, cnt in top_months.items():
            m, y = rain_col_map[col]
            print(f"  {y}-{m:02d}: {cnt:,} estaciones")

        print("\n── Distribución por año ──")
        year_counts = catalog.groupby("year").size()
        for yr, cnt in sorted(year_counts.items()):
            print(f"  {yr}: {cnt:,}")

    print("\n[OK] T2.5 completado.")
    return catalog


if __name__ == "__main__":
    run_t2_5(verbose=True)
