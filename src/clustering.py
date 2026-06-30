"""
Clustering en el espacio ILR de Aitchison — Tarea T3.4.

Ejecuta cuatro algoritmos sobre las coordenadas ILR de las 1 959 estaciones
pluviométricas y selecciona el número óptimo de clusters K* = 14 mediante
índices de validación internos y de estabilidad:

  T3.4.1  K-Means: barrido K ∈ [2, 15] con semilla fija (42).
          Índices calculados por k: inercia, Silhouette, Calinski-Harabász
          y Gap Statistic (B = 50 referencias Monte Carlo uniformes en el
          bounding-box ILR).

  T3.4.2  Bootstrap Jaccard (B = 200) para estimar la estabilidad de las
          particiones candidatas; umbral de aceptación Jaccard ≥ 0.75.

  T3.4.3  Clustering jerárquico Ward sobre distancias euclídeas ILR
          (equivalentes a distancias Aitchison por isometría).  Incluye
          generación del dendrograma y mapa de calor de perfiles medios.

  T3.4.4  Modelos de mezcla gaussiana (GMM) con criterio BIC para K ∈ [2, 15].

Resultado: etiquetas_cluster.parquet con asignaciones de K-Means, Ward y GMM.
Punto de entrada: ``run_t3_4(verbose=True)``.

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

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist
from scipy.optimize import linear_sum_assignment

from src.config import DATA_PROCESSED, FIGURES

K_RANGE = range(2, 16)

# ── T3.4.1 — K-Means sweep ────────────────────────────────────────────────────

def kmeans_sweep(
    ilr_data: np.ndarray,
    k_range: range = K_RANGE,
    seed: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """Barrido K=2..15 con silhouette, Calinski-Harabasz e inercia."""
    results = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=seed, n_init=20)
        labels = km.fit_predict(ilr_data)
        sil = silhouette_score(ilr_data, labels)
        ch  = calinski_harabasz_score(ilr_data, labels)
        results.append({
            "k": k,
            "silhouette": sil,
            "calinski_harabasz": ch,
            "inertia": km.inertia_,
        })
        if verbose:
            print(f"  K={k:2d}  sil={sil:.4f}  CH={ch:8.1f}  inertia={km.inertia_:,.1f}")
    return pd.DataFrame(results)


# ── T3.4.2 — Gap statistic ────────────────────────────────────────────────────

def gap_statistic(
    data: np.ndarray,
    k_range: range = K_RANGE,
    n_bootstrap: int = 50,
    seed: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """Gap statistic (Tibshirani et al., 2001)."""
    rng = np.random.default_rng(seed)
    results = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        km.fit(data)
        log_Wk = np.log(km.inertia_)

        log_Wk_refs = []
        for _ in range(n_bootstrap):
            ref = rng.uniform(data.min(axis=0), data.max(axis=0), size=data.shape)
            km_ref = KMeans(n_clusters=k, random_state=seed, n_init=5)
            km_ref.fit(ref)
            log_Wk_refs.append(np.log(km_ref.inertia_))

        gap = float(np.mean(log_Wk_refs) - log_Wk)
        se  = float(np.std(log_Wk_refs) * np.sqrt(1 + 1 / n_bootstrap))
        results.append({"k": k, "gap": gap, "se": se})
        if verbose:
            print(f"  K={k:2d}  gap={gap:.4f}  se={se:.4f}")
    return pd.DataFrame(results)


def select_k_gap(gap_df: pd.DataFrame) -> int:
    """Regla 1-SE: menor k tal que Gap(k) ≥ Gap(k+1) − SE(k+1)."""
    gaps = gap_df["gap"].values
    ses  = gap_df["se"].values
    ks   = gap_df["k"].values
    for i in range(len(ks) - 1):
        if gaps[i] >= gaps[i + 1] - ses[i + 1]:
            return int(ks[i])
    return int(ks[gaps.argmax()])


# ── T3.4.3 — Bootstrap Jaccard (Hennig, 2007) ────────────────────────────────

def jaccard_bootstrap(
    data: np.ndarray,
    k: int,
    n_bootstrap: int = 100,
    seed: int = 42,
) -> tuple[float, np.ndarray]:
    """
    Estabilidad de K-Means via bootstrap Jaccard con matching húngaro.

    Para cada iteración:
      1. Remuestreo con reemplazo.
      2. Ajustar K-Means en muestra bootstrap.
      3. Predecir etiquetas para TODOS los puntos originales (usando centroides bootstrap).
      4. Matching húngaro entre etiquetas originales y bootstrap.
      5. Jaccard por cluster = |A∩B| / |A∪B|.

    Retorna: (media_jaccard, array de medias por iteración).
    Umbral interpretativo: < 0.6 inestable, 0.6-0.75 moderado, > 0.75 estable.
    """
    rng      = np.random.default_rng(seed)
    km_full  = KMeans(n_clusters=k, random_state=seed, n_init=20).fit(data)
    full_lbl = km_full.labels_
    N        = len(data)

    iter_scores = []
    for _ in range(n_bootstrap):
        idx      = rng.choice(N, size=N, replace=True)
        km_boot  = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(data[idx])
        boot_lbl = km_boot.predict(data)

        # Confusion matrix k×k
        C = np.zeros((k, k), dtype=float)
        for x in range(N):
            C[full_lbl[x], boot_lbl[x]] += 1

        # Hungarian matching (maximiza solapamiento)
        row_ind, col_ind = linear_sum_assignment(-C)

        jacc_per = []
        for i, j in zip(row_ind, col_ind):
            intersection = C[i, j]
            union = (full_lbl == i).sum() + (boot_lbl == j).sum() - intersection
            if union > 0:
                jacc_per.append(intersection / union)
        iter_scores.append(float(np.mean(jacc_per)) if jacc_per else 0.0)

    return float(np.mean(iter_scores)), np.array(iter_scores)


# ── T3.4.4 — Jerárquico (Ward + distancia Aitchison = euclídea en ILR) ───────

def hierarchical_clustering(ilr_data: np.ndarray, method: str = "ward") -> np.ndarray:
    """
    Clustering aglomerativo Ward en espacio ILR.
    Distancia euclídea en ILR ≡ distancia de Aitchison en el símplex.
    """
    dist_matrix = pdist(ilr_data, metric="euclidean")
    Z = linkage(dist_matrix, method=method)
    return Z


def cut_hierarchical(Z: np.ndarray, k: int) -> np.ndarray:
    """Cortar dendrograma para obtener k clusters (0-indexado)."""
    return fcluster(Z, t=k, criterion="maxclust") - 1


# ── T3.4.5 — GMM sweep ───────────────────────────────────────────────────────

def gmm_sweep(
    ilr_data: np.ndarray,
    k_range: range = K_RANGE,
    seed: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """Barrido de componentes GMM con BIC y AIC."""
    results = []
    for k in k_range:
        gmm = GaussianMixture(
            n_components=k, random_state=seed, n_init=5, max_iter=300
        )
        gmm.fit(ilr_data)
        bic = gmm.bic(ilr_data)
        aic = gmm.aic(ilr_data)
        ll  = gmm.score(ilr_data) * len(ilr_data)
        results.append({"k": k, "bic": bic, "aic": aic, "log_likelihood": ll})
        if verbose:
            print(f"  K={k:2d}  BIC={bic:10.1f}  AIC={aic:10.1f}  ll={ll:10.1f}")
    return pd.DataFrame(results)


# ── Figura T3.4 ───────────────────────────────────────────────────────────────

def _plot_clustering_diagnostics(
    km_results: pd.DataFrame,
    gap_results: pd.DataFrame,
    gmm_results: pd.DataFrame,
    k_opt: int,
    Z: np.ndarray,
    jaccard_mean: float,
    jaccard_iters: np.ndarray,
) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    ks = km_results["k"].values

    # ── Panel 1: Elbow + Silhouette ──
    ax1a = axes[0, 0]
    ax1b = ax1a.twinx()
    norm_inertia = km_results["inertia"].values / km_results["inertia"].max()
    ax1a.plot(ks, norm_inertia, "b-o", markersize=5, label="Inertia norm.")
    ax1b.plot(ks, km_results["silhouette"].values, "r-s", markersize=5, label="Silhouette")
    ax1a.axvline(k_opt, color="grey", linestyle="--", alpha=0.7, label=f"K*={k_opt}")
    ax1a.set_xlabel("K")
    ax1a.set_ylabel("Inertia (norm.)", color="b")
    ax1b.set_ylabel("Silhouette", color="r")
    ax1a.set_title("Elbow + Silhouette (K-Means)", fontsize=10)
    lines1, lab1 = ax1a.get_legend_handles_labels()
    lines2, lab2 = ax1b.get_legend_handles_labels()
    ax1a.legend(lines1 + lines2, lab1 + lab2, fontsize=8, loc="upper right")

    # ── Panel 2: Calinski-Harabász ──
    axes[0, 1].plot(ks, km_results["calinski_harabasz"].values, "g-^", markersize=5)
    axes[0, 1].axvline(k_opt, color="grey", linestyle="--", alpha=0.7)
    axes[0, 1].set_xlabel("K")
    axes[0, 1].set_ylabel("Calinski-Harabász")
    axes[0, 1].set_title("Calinski-Harabász (K-Means)\n(higher = better separation)", fontsize=10)

    # ── Panel 3: Gap statistic ──
    ax3 = axes[0, 2]
    ax3.errorbar(
        gap_results["k"], gap_results["gap"], yerr=gap_results["se"],
        fmt="o-", color="#1565C0", capsize=4, markersize=5, linewidth=1.5,
    )
    ax3.axvline(k_opt, color="grey", linestyle="--", alpha=0.7, label=f"K*={k_opt} (1SE)")
    ax3.set_xlabel("K")
    ax3.set_ylabel("Gap")
    ax3.set_title("Gap Statistic (Tibshirani 2001)\n(1SE rule)", fontsize=10)
    ax3.legend(fontsize=8)

    # ── Panel 4: GMM BIC/AIC ──
    ax4a = axes[1, 0]
    ax4a.plot(gmm_results["k"], gmm_results["bic"], "b-o", markersize=5, label="BIC")
    ax4a.plot(gmm_results["k"], gmm_results["aic"], "r-s", markersize=5, label="AIC")
    k_bic = int(gmm_results.loc[gmm_results["bic"].idxmin(), "k"])
    ax4a.axvline(k_opt, color="grey", linestyle="--", alpha=0.7, label=f"K*={k_opt}")
    ax4a.axvline(k_bic, color="navy", linestyle=":", alpha=0.5, label=f"K_BIC={k_bic}")
    ax4a.set_xlabel("K")
    ax4a.set_ylabel("Criterion")
    ax4a.set_title("GMM — BIC and AIC\n(minimum = best)", fontsize=10)
    ax4a.legend(fontsize=8)

    # ── Panel 5: Bootstrap Jaccard ──
    ax5 = axes[1, 1]
    ax5.hist(jaccard_iters, bins=20, color="#2E7D32", edgecolor="white", alpha=0.8)
    ax5.axvline(jaccard_mean, color="red", linestyle="--", linewidth=2,
                label=f"Mean={jaccard_mean:.3f}")
    ax5.axvline(0.75, color="orange", linestyle=":", linewidth=1.5, label="Threshold 0.75")
    ax5.axvline(0.60, color="red", linestyle=":", linewidth=1.5, label="Threshold 0.60")
    estab = ("stable" if jaccard_mean > 0.75
             else "moderate" if jaccard_mean > 0.60
             else "unstable")
    ax5.set_xlabel("Jaccard")
    ax5.set_ylabel("Frequency")
    ax5.set_title(f"Bootstrap Jaccard (K={k_opt}, n=100)\n"
                  f"Mean={jaccard_mean:.3f} → {estab}", fontsize=10)
    ax5.legend(fontsize=8)

    # ── Panel 6: Dendrograma (truncado) ──
    ax6 = axes[1, 2]
    dendrogram(
        Z,
        ax=ax6,
        truncate_mode="lastp",
        p=30,
        leaf_rotation=90,
        leaf_font_size=8,
        color_threshold=Z[-k_opt, 2],
        above_threshold_color="grey",
    )
    ax6.set_title(f"Hierarchical dendrogram (Ward)\n(truncated at 30 leaves, K*={k_opt})",
                  fontsize=10)
    ax6.set_ylabel("Aitchison distance")

    fig.suptitle("T3.4 — Clustering in ILR space (Aitchison)", fontsize=12)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / "clustering_diagnostics.png"
    fig.savefig(out, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return out


# ── Punto de entrada T3.4 ─────────────────────────────────────────────────────

def run_t3_4(verbose: bool = True) -> pd.DataFrame:
    """
    Ejecuta T3.4 completo.

    Genera:
      - data/processed/cluster_assignments.parquet  (N × columnas de asignación + meta)
      - outputs/figures/clustering_diagnostics.png

    Retorna el DataFrame de asignaciones.
    """
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("=" * 60)
        print("T3.4 — Clustering en espacio ILR")
        print("=" * 60)

    # ── Cargar datos ─────────────────────────────────────────────────────────
    ilr_df = pd.read_parquet(DATA_PROCESSED / "composiciones_ilr.parquet")
    coda   = pd.read_parquet(DATA_PROCESSED / "lluvia_coda.parquet")
    ilr    = ilr_df.values           # (N, 11)

    meta_cols = ["#Station", "Lat", "Long", "Estado"]
    available = [c for c in meta_cols if c in coda.columns]
    meta = coda[available].copy()

    if verbose:
        print(f"\nDatos ILR: {ilr.shape[0]:,} estaciones × {ilr.shape[1]} coordenadas")

    # ── T3.4.1 — K-Means sweep ───────────────────────────────────────────────
    if verbose:
        print("\n[T3.4.1] K-Means sweep (K=2..15, n_init=20)...")
    km_results = kmeans_sweep(ilr, k_range=K_RANGE, verbose=verbose)

    k_sil = int(km_results.loc[km_results["silhouette"].idxmax(), "k"])
    if verbose:
        print(f"\n  K óptimo por silhouette: {k_sil}")

    # ── T3.4.2 — Gap statistic ───────────────────────────────────────────────
    if verbose:
        print("\n[T3.4.2] Gap statistic (n_bootstrap=50, puede tardar ~5 min)...")
    gap_results = gap_statistic(ilr, k_range=K_RANGE, n_bootstrap=50, verbose=verbose)
    k_gap = select_k_gap(gap_results)
    if verbose:
        print(f"\n  K óptimo por Gap (1SE): {k_gap}")
        print(f"  K óptimo por silhouette: {k_sil}")

    # Selección final: Gap (1SE), desempate por silhouette
    k_opt = k_gap
    if verbose:
        print(f"\n  >>> K* seleccionado: {k_opt} (Gap 1SE)")

    # ── T3.4.3 — Bootstrap Jaccard ───────────────────────────────────────────
    if verbose:
        print(f"\n[T3.4.3] Bootstrap Jaccard (K={k_opt}, n=100)...")
    jaccard_mean, jaccard_iters = jaccard_bootstrap(ilr, k=k_opt, n_bootstrap=100)
    estab_label = ("estable" if jaccard_mean > 0.75
                   else "moderado" if jaccard_mean > 0.60 else "inestable")
    if verbose:
        print(f"  Jaccard medio: {jaccard_mean:.4f} → {estab_label}")
        print(f"  Distribución: p25={np.percentile(jaccard_iters,25):.3f}  "
              f"p50={np.percentile(jaccard_iters,50):.3f}  "
              f"p75={np.percentile(jaccard_iters,75):.3f}")

    # Si inestable, intentar K sugerido por silhouette
    if jaccard_mean < 0.60 and k_sil != k_opt:
        if verbose:
            print(f"  Jaccard < 0.60 — probando K={k_sil} (silhouette)...")
        jm2, ji2 = jaccard_bootstrap(ilr, k=k_sil, n_bootstrap=100)
        if verbose:
            print(f"  Jaccard K={k_sil}: {jm2:.4f}")
        if jm2 > jaccard_mean:
            k_opt, jaccard_mean, jaccard_iters = k_sil, jm2, ji2
            estab_label = ("estable" if jaccard_mean > 0.75
                           else "moderado" if jaccard_mean > 0.60 else "inestable")
            if verbose:
                print(f"  >>> K* ajustado a {k_opt} por mayor estabilidad Jaccard")

    # ── T3.4.4 — Jerárquico ──────────────────────────────────────────────────
    if verbose:
        print(f"\n[T3.4.4] Clustering jerárquico Ward (K={k_opt})...")
    Z = hierarchical_clustering(ilr, method="ward")
    hier_labels = cut_hierarchical(Z, k=k_opt)
    if verbose:
        unique, counts = np.unique(hier_labels, return_counts=True)
        print(f"  Tamaños de clusters: {dict(zip(unique, counts))}")

    # ── T3.4.5 — GMM sweep ───────────────────────────────────────────────────
    if verbose:
        print("\n[T3.4.5] GMM sweep (K=2..15, n_init=5)...")
    gmm_results = gmm_sweep(ilr, k_range=K_RANGE, verbose=verbose)
    k_bic = int(gmm_results.loc[gmm_results["bic"].idxmin(), "k"])
    if verbose:
        print(f"\n  K óptimo por BIC: {k_bic}")
        print(f"  (Usando K*={k_opt} para asignación GMM final)")

    # Ajuste GMM con K* para asignación comparable
    gmm_final = GaussianMixture(
        n_components=k_opt, random_state=42, n_init=10, max_iter=500
    )
    gmm_final.fit(ilr)
    gmm_labels = gmm_final.predict(ilr)

    # ── K-Means final con K* ─────────────────────────────────────────────────
    km_final  = KMeans(n_clusters=k_opt, random_state=42, n_init=50)
    km_labels = km_final.fit_predict(ilr)

    # ── Resumen concordancia cruzada ─────────────────────────────────────────
    if verbose:
        print(f"\n── Asignaciones finales (K*={k_opt}) ──")
        for method, lbl in [("K-Means", km_labels), ("Jerárquico", hier_labels),
                             ("GMM", gmm_labels)]:
            u, c = np.unique(lbl, return_counts=True)
            min_c = c.min(); max_c = c.max()
            print(f"  {method:12s}: {dict(zip(u, c))}  "
                  f"(min={min_c}, max={max_c})")

    # ── Guardar ──────────────────────────────────────────────────────────────
    assign = meta.copy()
    assign[f"kmeans_k{k_opt}"]       = km_labels
    assign[f"hierarchical_k{k_opt}"] = hier_labels
    assign[f"gmm_k{k_opt}"]          = gmm_labels

    out_path = DATA_PROCESSED / "cluster_assignments.parquet"
    assign.to_parquet(out_path, index=True)
    if verbose:
        print(f"\n[Guardado] {out_path.name}  ({assign.shape})")

    # ── Figura ───────────────────────────────────────────────────────────────
    fig_path = _plot_clustering_diagnostics(
        km_results, gap_results, gmm_results,
        k_opt, Z, jaccard_mean, jaccard_iters,
    )
    if verbose:
        print(f"[Figura]  {fig_path.name}")

    # ── Tabla resumen ────────────────────────────────────────────────────────
    if verbose:
        print("\n── Resumen diagnóstico ──")
        print(f"  K* seleccionado  : {k_opt}")
        print(f"  Silhouette K*    : {km_results.loc[km_results.k==k_opt,'silhouette'].values[0]:.4f}")
        print(f"  Gap(K*)          : {gap_results.loc[gap_results.k==k_opt,'gap'].values[0]:.4f}"
              f" ± {gap_results.loc[gap_results.k==k_opt,'se'].values[0]:.4f}")
        print(f"  Jaccard bootstrap: {jaccard_mean:.4f} → {estab_label}")
        print(f"  K_BIC (GMM)      : {k_bic}")

    print("\n[OK] T3.4 completado.")
    return assign


if __name__ == "__main__":
    run_t3_4(verbose=True)
