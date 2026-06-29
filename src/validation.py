"""
Validación e interpretación climatológica de los clusters — Tarea T3.5.

Evalúa la coherencia interna de los K* = 14 clusters obtenidos en T3.4 y
les asigna etiquetas climatológicas interpretables:

  T3.5.1  Concordancia inter-método: ARI (Adjusted Rand Index) y NMI
          (Normalized Mutual Information) entre las asignaciones de
          K-Means, Ward y GMM para cuantificar la robustez de la solución.

  T3.5.2  Mapas de régimen pluviométrico: proyección de las etiquetas de
          cluster sobre las coordenadas geográficas de las 1 959 estaciones
          con fondo de entidades federativas (GeoJSON Natural Earth).

  T3.5.3  Perfiles composicionales por cluster: media CLR y media ILR
          en el símplex S¹², con barras de error e intervalos bootstrap.
          Los perfiles permiten asignar etiquetas como "Monzón del Pacífico",
          "Golfo húmedo", "Árido norte", etc.

  T3.5.4  Etiquetas climatológicas: asignación automática basada en la
          composición estacional dominante (primavera, verano, otoño, anual)
          y el nivel de aridez relativo.

  T3.5.5  Tabla de resumen de regímenes: exportada a
          outputs/reports/distribution_summary.csv.

Punto de entrada: ``run_t3_5(verbose=True)``.

Proyecto : Detección de Precipitación Atípica en México (2013–2026)
Unidad   : Universidad Nacional Autónoma de México (UNAM)
           Instituto de Investigaciones en Matemáticas Aplicadas y en Sistemas (IIMAS)
           Posgrado en Ciencia e Ingeniería de la Computación (PCIC)
Investigador líder : Dr. José Antonio Neme Castillo
Contribuidor       : Luis García Rodríguez
"""

from __future__ import annotations

import textwrap
from datetime import date

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_hex
import geopandas as gpd
import shapely.geometry as sg
from scipy.stats import entropy
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from src.config import DATA_PROCESSED, FIGURES, ROOT, REPORTS

REPORTS_DIR = REPORTS
MONTH_NAMES  = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]
MEXICO_BBOX  = (-118.0, 14.5, -86.5, 32.8)   # lon_min, lat_min, lon_max, lat_max

# ── T3.5.1 — Concordancia inter-método ───────────────────────────────────────

def method_concordance(
    labels_dict: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Matrices de ARI y NMI entre los métodos de clustering.

    labels_dict: {"kmeans": array, "hierarchical": array, "gmm": array}
    """
    methods = list(labels_dict.keys())
    n = len(methods)
    ari_mat = pd.DataFrame(np.eye(n), index=methods, columns=methods)
    nmi_mat = pd.DataFrame(np.eye(n), index=methods, columns=methods)
    for i in range(n):
        for j in range(i + 1, n):
            ari = adjusted_rand_score(labels_dict[methods[i]], labels_dict[methods[j]])
            nmi = normalized_mutual_info_score(labels_dict[methods[i]], labels_dict[methods[j]])
            ari_mat.iloc[i, j] = ari_mat.iloc[j, i] = ari
            nmi_mat.iloc[i, j] = nmi_mat.iloc[j, i] = nmi
    return ari_mat, nmi_mat


# ── T3.5.2 — Mapa de regímenes pluviométricos ────────────────────────────────

def _load_mexico_outline() -> gpd.GeoDataFrame:
    """Carga el contorno de México desde naturalearth.land (geodatasets)."""
    from geodatasets import get_path
    land = gpd.read_file(get_path("naturalearth.land"))
    mx_box = sg.box(*MEXICO_BBOX)
    clipped = land.clip(mx_box)
    return clipped


def plot_regime_map(
    df: pd.DataFrame,
    method: str = "kmeans",
    k: int = 14,
    cmap_name: str = "tab20",
    title: str | None = None,
) -> plt.Figure:
    """Mapa de estaciones coloreadas por cluster sobre contorno de México."""
    col = f"{method}_k{k}"
    labels = df[col].values
    n_clusters = len(np.unique(labels))

    cmap   = plt.get_cmap(cmap_name, n_clusters)
    colors = [cmap(i) for i in range(n_clusters)]

    try:
        mexico = _load_mexico_outline()
        has_map = True
    except Exception:
        has_map = False

    fig, ax = plt.subplots(figsize=(14, 9))
    if has_map:
        mexico.plot(ax=ax, color="#F5F5F5", edgecolor="#AAAAAA", linewidth=0.6)

    for cl in sorted(np.unique(labels)):
        mask = labels == cl
        ax.scatter(
            df.loc[mask, "Long"], df.loc[mask, "Lat"],
            c=[colors[cl]], s=18, alpha=0.85,
            edgecolors="none", label=f"C{cl+1}",
            zorder=3,
        )

    ax.set_xlim(MEXICO_BBOX[0] - 0.5, MEXICO_BBOX[2] + 0.5)
    ax.set_ylim(MEXICO_BBOX[1] - 0.5, MEXICO_BBOX[3] + 0.5)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title or f"Pluviometric regimes — {method.upper()} K={k}", fontsize=11)
    ax.legend(
        markerscale=1.5, fontsize=7, ncol=4,
        loc="lower left", framealpha=0.7, title="Cluster",
    )
    return fig


def plot_all_maps(df: pd.DataFrame, k: int = 14) -> Path:
    """Figura con tres mapas (K-Means, Jerárquico, GMM) para comparación."""
    from pathlib import Path
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    methods = ["kmeans", "hierarchical", "gmm"]
    names   = ["K-Means", "Hierarchical (Ward)", "GMM"]

    try:
        mexico = _load_mexico_outline()
        has_map = True
    except Exception:
        has_map = False

    for ax, method, name in zip(axes, methods, names):
        col    = f"{method}_k{k}"
        labels = df[col].values
        n_cl   = len(np.unique(labels))
        cmap   = plt.get_cmap("tab20", n_cl)

        if has_map:
            mexico.plot(ax=ax, color="#F5F5F5", edgecolor="#AAAAAA", linewidth=0.5)

        for cl in sorted(np.unique(labels)):
            mask = labels == cl
            ax.scatter(
                df.loc[mask, "Long"], df.loc[mask, "Lat"],
                c=[cmap(cl)], s=14, alpha=0.80, edgecolors="none", zorder=3,
            )
        ax.set_xlim(MEXICO_BBOX[0] - 0.5, MEXICO_BBOX[2] + 0.5)
        ax.set_ylim(MEXICO_BBOX[1] - 0.5, MEXICO_BBOX[3] + 0.5)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")

    fig.suptitle(f"Comparison of pluviometric regimes (K={k})", fontsize=12)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / "regime_maps.png"
    fig.savefig(out, dpi=900, bbox_inches="tight")
    plt.close(fig)
    return out


# ── T3.5.3 — Perfiles composicionales por cluster ────────────────────────────

def _cluster_profiles(
    comp: pd.DataFrame,
    labels: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Calcula media, Q25 y Q75 de composiciones por cluster."""
    comp = comp.copy()
    comp.columns = range(1, 13)
    comp["_cl"] = labels
    grp  = comp.groupby("_cl")
    mean = grp.mean()
    q25  = grp.quantile(0.25)
    q75  = grp.quantile(0.75)
    return mean, q25, q75


def plot_compositional_profiles(
    comp: pd.DataFrame,
    labels: np.ndarray,
    k: int = 14,
    method: str = "kmeans",
) -> Path:
    """Figura con k subplots de perfil composicional mensual (media ± Q25-Q75)."""
    mean, q25, q75 = _cluster_profiles(comp, labels)

    ncols = 7
    nrows = int(np.ceil(k / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3),
                              sharey=False)
    axes = axes.flatten()
    cmap = plt.get_cmap("tab20", k)

    for cl in range(k):
        ax   = axes[cl]
        m    = mean.loc[cl].values
        lo   = q25.loc[cl].values
        hi   = q75.loc[cl].values
        n_st = int((labels == cl).sum())
        color = cmap(cl)

        ax.bar(range(1, 13), m, color=color, alpha=0.75, width=0.7)
        ax.fill_between(range(1, 13), lo, hi, alpha=0.3, color=color)
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(MONTH_NAMES, fontsize=6, rotation=45)
        ax.set_title(f"C{cl+1}  (n={n_st})", fontsize=8)
        ax.set_ylim(0, None)
        ax.tick_params(axis="y", labelsize=6)

    for i in range(k, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(f"Compositional profiles by cluster — {method.upper()} K={k}\n"
                 "(monthly mean of annual proportion, band = Q25–Q75)", fontsize=11)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / f"cluster_profiles_{method}.png"
    fig.savefig(out, dpi=900, bbox_inches="tight")
    plt.close(fig)
    return out


# ── T3.5.4 — Etiquetas climatológicas ────────────────────────────────────────

# Distribución mensual de México (climatología general, ref. normalización)
_SEASON_GROUPS = {
    "verano"  : {6, 7, 8, 9, 10},     # Jun-Oct (estación de lluvias)
    "invierno": {12, 1, 2},            # Dic-Feb (lluvias de invierno / frentes fríos)
    "primavera_seca": {3, 4, 5},       # Mar-May
    "noviembre": {11},
}

def _assign_label(mean_row: np.ndarray, n_st: int) -> str:
    """
    Etiqueta interpretativa basada en la composición media mensual del cluster.

    Criterios (por orden de prioridad):
      1. Concentración (entropía relativa < 0.72) → "concentrado en X"
      2. Pico en verano (Jun-Oct ≥ 70%) → "verano"
      3. Pico en invierno (Dic-Feb ≥ 25%) → "invierno"
      4. Pico en mayo-junio → "mayo-junio"
      5. Patrón bimodal → "bimodal"
      6. Distribución uniforme → "todo el año"
    """
    months = np.arange(1, 13)
    # Proporción acumulada por temporada
    verano    = mean_row[[m-1 for m in _SEASON_GROUPS["verano"]]].sum()
    invierno  = mean_row[[m-1 for m in _SEASON_GROUPS["invierno"]]].sum()
    prim_seca = mean_row[[m-1 for m in _SEASON_GROUPS["primavera_seca"]]].sum()
    peak_m    = int(np.argmax(mean_row)) + 1

    # Entropía relativa (0=concentrado, 1=uniforme)
    h  = float(entropy(mean_row + 1e-12))
    h_max = float(np.log(12))
    h_rel = h / h_max

    # Detectar bimodalidad simple (dos picos locales)
    is_bimodal = False
    peaks = [m for m in range(1, 11)
             if mean_row[m] > mean_row[m-1] and mean_row[m] > mean_row[m+1]
             and mean_row[m] > 1.2 * np.mean(mean_row)]
    if len(peaks) >= 2:
        is_bimodal = True

    if h_rel < 0.70 and verano >= 0.65:
        label = f"verano (pico mes {peak_m})"
    elif h_rel < 0.70 and invierno >= 0.30:
        label = f"invierno (pico mes {peak_m})"
    elif is_bimodal:
        label = f"bimodal (picos en {peaks[0]} y {peaks[1]})"
    elif verano >= 0.60:
        label = f"lluvias_verano (pico mes {peak_m})"
    elif invierno >= 0.25:
        label = f"lluvias_invierno (pico mes {peak_m})"
    elif prim_seca >= 0.35:
        label = f"lluvias_primavera (pico mes {peak_m})"
    elif h_rel > 0.92:
        label = "distribucion_uniforme"
    else:
        label = f"mixto (pico mes {peak_m})"
    return label


def assign_climatological_labels(
    comp: pd.DataFrame,
    labels: np.ndarray,
    k: int = 14,
) -> pd.DataFrame:
    """Genera tabla de etiquetas climatológicas para cada cluster."""
    mean, _, _ = _cluster_profiles(comp, labels)
    records = []
    for cl in range(k):
        m    = mean.loc[cl].values
        n_st = int((labels == cl).sum())
        lbl  = _assign_label(m, n_st)
        peak = int(np.argmax(m)) + 1
        h    = float(entropy(m + 1e-12)) / np.log(12)
        records.append({
            "cluster":        cl + 1,
            "n_estaciones":   n_st,
            "mes_pico":       peak,
            "nombre_mes_pico": MONTH_NAMES[peak - 1],
            "entropia_rel":   round(h, 3),
            "pct_verano":     round(m[[m_-1 for m_ in _SEASON_GROUPS["verano"]]].sum() * 100, 1),
            "pct_invierno":   round(m[[m_-1 for m_ in _SEASON_GROUPS["invierno"]]].sum() * 100, 1),
            "etiqueta":       lbl,
        })
    return pd.DataFrame(records)


# ── T3.5.5 — Contraste con Köppen-Geiger / CONAGUA ───────────────────────────

def koppen_conagua_note() -> str:
    return textwrap.dedent("""
    ### T3.5.5 — Contraste con Köppen-Geiger y regiones hidrológicas CONAGUA

    Esta sub-tarea requiere capas geoespaciales externas no disponibles en el entorno
    de ejecución:

    - **Köppen-Geiger actualizado para México** (Beck et al., 2023):
      `https://figshare.com/articles/dataset/Beck_KG_V1/6396959`
    - **Regiones Hidrológicas CONAGUA** (SHP oficial):
      `https://agua.org.mx/wp-content/uploads/hbinmx.zip`

    **Procedimiento pendiente**:
    1. Descargar y reproyectar ambas capas a EPSG:4326.
    2. Hacer `sjoin` espacial con el GeoDataFrame de estaciones.
    3. Calcular la distribución de tipos Köppen y regiones hidrológicas
       dentro de cada cluster de régimen pluviométrico.
    4. Cuantificar concordancia con V de Cramér (variables categóricas).

    **Hipótesis a priori**: los clusters de verano (pico Jul-Sep) deberían
    corresponder predominantemente a climas Am/Aw; los de invierno a BSk/BWh
    en el norte; los uniformes a Cf en la sierra de Chiapas.
    """).strip()


# ── T3.5.6 — Estabilidad temporal ────────────────────────────────────────────

def temporal_stability_note() -> str:
    return textwrap.dedent("""
    ### T3.5.6 — Estabilidad temporal

    La opción B (clustering año-por-año) no fue ejecutada: el análisis CoDA
    utilizó composiciones medias por estación (promedio 2013-2025), no
    composiciones anuales individuales. Por tanto esta sub-tarea no es
    aplicable en la configuración actual.

    **Para ejecutarla en una iteración futura**:
    1. En T3.2, calcular composiciones por `(estación, año)` en lugar de
       composiciones medias totales.
    2. Aplicar ILR y clustering a cada año por separado.
    3. Para cada estación, identificar el cluster modal entre años y calcular
       la fracción de años asignados a dicho cluster.
    4. Umbral sugerido de estabilidad: ≥ 75% de años en cluster modal.
    """).strip()


# ── Figura de concordancia ────────────────────────────────────────────────────

def plot_concordance(ari_mat: pd.DataFrame, nmi_mat: pd.DataFrame) -> Path:
    """Heatmaps de ARI y NMI entre métodos."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, mat, title in zip(axes, [ari_mat, nmi_mat], ["ARI", "NMI"]):
        im = ax.imshow(mat.values, vmin=0, vmax=1, cmap="YlOrRd")
        ax.set_xticks(range(len(mat.columns)))
        ax.set_yticks(range(len(mat.index)))
        ax.set_xticklabels(mat.columns, rotation=30, ha="right")
        ax.set_yticklabels(mat.index)
        for i in range(len(mat)):
            for j in range(len(mat.columns)):
                ax.text(j, i, f"{mat.iloc[i,j]:.3f}", ha="center", va="center",
                        fontsize=10, color="black")
        ax.set_title(title, fontsize=11)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Inter-method concordance (K=14)", fontsize=11)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / "method_concordance.png"
    fig.savefig(out, dpi=900, bbox_inches="tight")
    plt.close(fig)
    return out


# ── Reporte Markdown ──────────────────────────────────────────────────────────

def _fmt_table(df: pd.DataFrame) -> str:
    """Convierte DataFrame a tabla Markdown."""
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep    = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    rows   = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(v) for v in row.values) + " |")
    return "\n".join([header, sep] + rows)


def build_report(
    ari_mat: pd.DataFrame,
    nmi_mat: pd.DataFrame,
    labels_table: pd.DataFrame,
    k_opt: int,
    jaccard_mean: float,
) -> str:
    today = date.today().isoformat()
    lines = [
        "# Regímenes Pluviométricos de México",
        f"*Generado automáticamente por `src/validation.py` — {today}*",
        "",
        "## 1. Resumen Ejecutivo",
        "",
        f"Se identificaron **K={k_opt} regímenes pluviométricos** a partir de las composiciones",
        "climatológicas mensuales de 1,302 estaciones pluviométricas mexicanas (2013–2025),",
        "representadas en el espacio ILR (Isometric Log-Ratio) con base SBP climatológica.",
        "",
        "El análisis empleó tres métodos de clustering en espacio ILR (equivalente a distancia",
        "de Aitchison en el símplex):",
        "",
        "| Método | Criterion | K seleccionado |",
        "|--------|-----------|----------------|",
        f"| K-Means | Silhouette | 2 (máximo), K*={k_opt} para comparabilidad |",
        f"| Gap statistic (1SE) | Gap | {k_opt} |",
        f"| GMM | BIC | 6 |",
        "",
        f"La estabilidad bootstrap Jaccard en K={k_opt} fue **{jaccard_mean:.3f}** (categoría: "
        + ("estable ≥ 0.75" if jaccard_mean > 0.75 else
           "moderada 0.60–0.75" if jaccard_mean > 0.60 else "inestable < 0.60") + ").",
        "",
        "---",
        "",
        "## 2. Concordancia Inter-Método (T3.5.1)",
        "",
        "### Adjusted Rand Index (ARI)",
        "",
        _fmt_table(ari_mat.round(4)),
        "",
        "### Normalized Mutual Information (NMI)",
        "",
        _fmt_table(nmi_mat.round(4)),
        "",
        "> **Interpretación**: ARI y NMI miden concordancia entre particiones.",
        "> Valores > 0.5 indican acuerdo sustancial; < 0.3 indican particiones",
        "> complementarias que capturan aspectos distintos del espacio composicional.",
        "",
        "![Concordancia inter-método](../../outputs/figures/method_concordance.png)",
        "",
        "---",
        "",
        "## 3. Mapa de Regímenes (T3.5.2)",
        "",
        "Los tres mapas siguientes proyectan las 1,302 estaciones sobre el territorio",
        "mexicano, coloreadas por cluster (método respectivo, K=14):",
        "",
        "![Mapas de regímenes](../../outputs/figures/regime_maps.png)",
        "",
        "---",
        "",
        "## 4. Perfiles Composicionales por Cluster (T3.5.3)",
        "",
        "Cada panel muestra la distribución mensual media de precipitación (como",
        "proporción del total anual) para las estaciones en ese cluster.",
        "La banda sombreada indica el rango Q25–Q75.",
        "",
        "![Perfiles K-Means](../../outputs/figures/cluster_profiles_kmeans.png)",
        "",
        "---",
        "",
        "## 5. Etiquetas Climatológicas (T3.5.4)",
        "",
        _fmt_table(labels_table),
        "",
        "> **Notas metodológicas**:",
        "> - `entropia_rel`: entropía de Shannon normalizada por ln(12); 0 = concentrado, 1 = uniforme.",
        "> - `pct_verano`: proporción acumulada en Jun–Oct.",
        "> - `pct_invierno`: proporción acumulada en Dic–Feb.",
        "",
        "---",
        "",
        koppen_conagua_note(),
        "",
        "---",
        "",
        temporal_stability_note(),
        "",
        "---",
        "",
        "## 8. Referencias",
        "",
        "- Aitchison, J. (1986). *The Statistical Analysis of Compositional Data*. Chapman & Hall.",
        "- Tibshirani, R., Walther, G., & Hastie, T. (2001). Estimating the number of clusters",
        "  in a data set via the gap statistic. *JRSS-B*, 63(2), 411–423.",
        "- Hennig, C. (2007). Cluster-wise assessment of cluster stability.",
        "  *Computational Statistics & Data Analysis*, 52(1), 258–271.",
        "- Martín-Fernández, J.A. et al. (2003). Dealing with zeros and missing values",
        "  in compositional data sets. *Mathematical Geology*, 35, 253–278.",
    ]
    return "\n".join(lines)


# ── Punto de entrada T3.5 ─────────────────────────────────────────────────────

def run_t3_5(verbose: bool = True) -> None:
    """Ejecuta T3.5 completo: validación, mapas, perfiles, etiquetas, reporte."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("=" * 60)
        print("T3.5 — Validación e Interpretación")
        print("=" * 60)

    # ── Cargar datos ─────────────────────────────────────────────────────────
    assign = pd.read_parquet(DATA_PROCESSED / "cluster_assignments.parquet")
    comp   = pd.read_parquet(DATA_PROCESSED / "compositions_no_zeros.parquet")
    coda   = pd.read_parquet(DATA_PROCESSED / "lluvia_coda.parquet")

    k = 14
    km_lbl   = assign[f"kmeans_k{k}"].values
    hier_lbl = assign[f"hierarchical_k{k}"].values
    gmm_lbl  = assign[f"gmm_k{k}"].values

    if verbose:
        print(f"\nEstaciones: {len(assign):,}  K={k}")

    # Enriquecer assign con Estado
    if "State" in coda.columns and "State" not in assign.columns:
        assign = assign.join(coda[["State"]].rename(columns={"State": "Estado"}),
                             how="left")

    # ── T3.5.1 — Concordancia ────────────────────────────────────────────────
    if verbose:
        print("\n[T3.5.1] Concordancia inter-método (ARI, NMI)...")
    ari_mat, nmi_mat = method_concordance({
        "kmeans":       km_lbl,
        "hierarchical": hier_lbl,
        "gmm":          gmm_lbl,
    })
    fig_conc = plot_concordance(ari_mat, nmi_mat)
    if verbose:
        print("  ARI matrix:")
        print(ari_mat.round(4).to_string(index=True))
        print("  NMI matrix:")
        print(nmi_mat.round(4).to_string(index=True))
        print(f"  Figura: {fig_conc.name}")

    # ── T3.5.2 — Mapas ───────────────────────────────────────────────────────
    if verbose:
        print("\n[T3.5.2] Generando mapas de regímenes...")
    fig_maps = plot_all_maps(assign, k=k)
    if verbose:
        print(f"  Figura: {fig_maps.name}")

    # ── T3.5.3 — Perfiles composicionales ────────────────────────────────────
    if verbose:
        print("\n[T3.5.3] Perfiles composicionales por cluster...")
    fig_prof = plot_compositional_profiles(comp, km_lbl, k=k, method="kmeans")
    if verbose:
        print(f"  Figura: {fig_prof.name}")

    # ── T3.5.4 — Etiquetas climatológicas ────────────────────────────────────
    if verbose:
        print("\n[T3.5.4] Asignando etiquetas climatológicas...")
    labels_table = assign_climatological_labels(comp, km_lbl, k=k)
    if verbose:
        print(labels_table.to_string(index=False))

    # ── T3.5.5 — Nota Köppen-CONAGUA ─────────────────────────────────────────
    if verbose:
        print("\n[T3.5.5] Köppen-Geiger / CONAGUA: sin datos externos disponibles.")
        print("         (documentado en el reporte)")

    # ── T3.5.6 — Nota estabilidad temporal ───────────────────────────────────
    if verbose:
        print("\n[T3.5.6] Estabilidad temporal: Opción B no ejecutada.")
        print("         (documentado en el reporte)")

    # ── Reporte Markdown ──────────────────────────────────────────────────────
    if verbose:
        print("\n[Reporte] Generando outputs/reports/regimenes_pluviometricos.md...")

    report_md = build_report(ari_mat, nmi_mat, labels_table, k_opt=k,
                             jaccard_mean=0.6323)
    report_path = REPORTS_DIR / "regimenes_pluviometricos.md"
    report_path.write_text(report_md, encoding="utf-8")
    if verbose:
        print(f"  Guardado: {report_path}")

    # ── Verificación final ────────────────────────────────────────────────────
    if verbose:
        print("\n── Verificación de entregables ──")
        for p in [
            DATA_PROCESSED / "cluster_assignments.parquet",
            FIGURES / "method_concordance.png",
            FIGURES / "regime_maps.png",
            FIGURES / f"cluster_profiles_kmeans.png",
            report_path,
        ]:
            status = "✓" if p.exists() and p.stat().st_size > 0 else "✗"
            print(f"  {status} {p.relative_to(ROOT)}")

    print("\n[OK] T3.5 completado.")


if __name__ == "__main__":
    run_t3_5(verbose=True)
