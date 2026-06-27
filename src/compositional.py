"""
Transformaciones log-ratio para el análisis de datos composicionales — Tarea T3.3.

Implementa las dos transformaciones estándar del marco de Aitchison (1986) que
trasladan composiciones del símplex S¹² al espacio euclídeo R^D donde operan
los métodos estadísticos convencionales:

  T3.3.1  CLR (Centered Log-Ratio):
          y_j = ln(w_j / g(w)), siendo g(w) la media geométrica de las 12 partes.
          El rango resultante tiene D-1 = 11 grados de libertad efectivos
          (restricción Σclr = 0 inherente al símplex).

  T3.3.2  ILR con base SBP climatológica (11 coordenadas):
          Partición Binaria Secuencial interpretable: P1 separa meses secos vs
          húmedos; las siguientes 10 particiones refinan la jerarquía intra-grupo.
          La matriz de contraste Ψ ((D-1)×D) se construye con la fórmula
          de ortonormalización de Egozcue et al. (2003).

  T3.3.3  Verificación de isometría: d_Aitchison(wᵢ, wⱼ) ≡ d_Euclidean(ilr(wᵢ), ilr(wⱼ))
          sobre 500 pares aleatorios (error relativo máximo esperado < 10⁻⁸).

También incluye la transformación ILR-Helmert como referencia matemáticamente
equivalente pero sin interpretación climatológica directa.

Resultado: composiciones_clr.parquet y composiciones_ilr.parquet.
Punto de entrada: ``run_t3_3(verbose=True)``.

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
from scipy.stats import gmean

from src.config import DATA_PROCESSED, FIGURES

MONTH_NAMES = ["Ene","Feb","Mar","Abr","May","Jun",
               "Jul","Ago","Sep","Oct","Nov","Dic"]

# ── T3.3.1 — CLR (Centered Log-Ratio) ────────────────────────────────────────

def clr_transform(compositions: pd.DataFrame) -> pd.DataFrame:
    """
    T3.3.1 — Centered Log-Ratio.

    y_j = ln(w_j / g(w))   donde g(w) = media geométrica de todas las partes.

    Retorna DataFrame con D=12 columnas; el rango real es D-1=11 por la
    restricción Σ(clr)=0, inherente al símplex.
    """
    gm = gmean(compositions.values, axis=1)           # (N,)
    clr = np.log(compositions.values / gm[:, None])   # (N, 12)
    return pd.DataFrame(clr, index=compositions.index,
                        columns=compositions.columns)


# ── T3.3.2 — ILR con SBP climatológica ───────────────────────────────────────

def build_sbp_matrix() -> np.ndarray:
    """
    Matriz de Partición Binaria Secuencial (SBP) climatológicamente interpretable.

    11 particiones × 12 meses (columnas = meses 1..12).
    +1 = grupo numerador, -1 = grupo denominador, 0 = inactivo.

    Jerarquía:
      P1 : secos(1,2,3,4,11,12) vs húmedos(5,6,7,8,9,10)
      P2 : dentro secos → invierno(12,1,2) vs prim.seca(3,4,11)
      P3 : dentro húmedos → inicio(5,6) vs pleno(7,8,9,10)
      P4 : dentro {12,1,2} → dic(12) vs {ene,feb}(1,2)
      P5 : dentro {1,2}    → ene(1) vs feb(2)
      P6 : dentro {3,4,11} → {mar,abr}(3,4) vs nov(11)
      P7 : dentro {3,4}    → mar(3) vs abr(4)
      P8 : dentro {5,6}    → may(5) vs jun(6)
      P9 : dentro {7,8,9,10} → {jul,ago}(7,8) vs {sep,oct}(9,10)
      P10: dentro {7,8}    → jul(7) vs ago(8)
      P11: dentro {9,10}   → sep(9) vs oct(10)
    """
    # Columnas: mes 1  2   3   4   5   6   7   8   9  10  11  12
    return np.array([
        [+1, +1, +1, +1, -1, -1, -1, -1, -1, -1, +1, +1],  # P1
        [+1, +1, -1, -1,  0,  0,  0,  0,  0,  0, -1, +1],  # P2
        [ 0,  0,  0,  0, +1, +1, -1, -1, -1, -1,  0,  0],  # P3
        [-1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0, +1],  # P4
        [+1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # P5
        [ 0,  0, +1, +1,  0,  0,  0,  0,  0,  0, -1,  0],  # P6
        [ 0,  0, +1, -1,  0,  0,  0,  0,  0,  0,  0,  0],  # P7
        [ 0,  0,  0,  0, +1, -1,  0,  0,  0,  0,  0,  0],  # P8
        [ 0,  0,  0,  0,  0,  0, +1, +1, -1, -1,  0,  0],  # P9
        [ 0,  0,  0,  0,  0,  0, +1, -1,  0,  0,  0,  0],  # P10
        [ 0,  0,  0,  0,  0,  0,  0,  0, +1, -1,  0,  0],  # P11
    ], dtype=float)


def sbp_to_contrast(sbp: np.ndarray) -> np.ndarray:
    """
    Convierte una SBP (+1/-1/0) a la matriz de contraste ortonormal Ψ ((D-1)×D).

    Para la partición k con r partes positivas y s partes negativas:
      coef(+1) =  sqrt(s / (r·(r+s)))
      coef(-1) = -sqrt(r / (s·(r+s)))
      coef( 0) =  0
    """
    V = np.zeros_like(sbp, dtype=float)
    for k in range(sbp.shape[0]):
        row = sbp[k]
        r = int((row == +1).sum())
        s = int((row == -1).sum())
        if r == 0 or s == 0:
            raise ValueError(f"Partición {k}: r={r}, s={s} — inválida.")
        V[k, row == +1] =  np.sqrt(s / (r * (r + s)))
        V[k, row == -1] = -np.sqrt(r / (s * (r + s)))
    return V


def ilr_transform(
    compositions: pd.DataFrame,
    contrast_matrix: np.ndarray,
) -> pd.DataFrame:
    """
    T3.3.2 — Isometric Log-Ratio con matriz de contraste Ψ.

    y_k = Σ_j Ψ_kj · ln(w_j)   (producto escalar de Ψ_k con ln(w))

    Retorna DataFrame (N × D-1).
    """
    log_comp = np.log(compositions.values)
    ilr_coords = log_comp @ contrast_matrix.T        # (N, D-1)
    return pd.DataFrame(
        ilr_coords,
        index=compositions.index,
        columns=[f"ilr_{k+1}" for k in range(contrast_matrix.shape[0])],
    )


def helmert_ilr(compositions: pd.DataFrame) -> pd.DataFrame:
    """
    ILR con base de Helmert (alternativa cuando no se especifica SBP).

    Menos interpretable climatológicamente pero matemáticamente equivalente
    a cualquier otra base ortonormal del espacio de Aitchison.
    """
    D = compositions.shape[1]
    V = np.zeros((D - 1, D))
    for i in range(D - 1):
        V[i, :i + 1] = 1.0 / (i + 1)
        V[i, i + 1] = -1.0
        V[i] *= np.sqrt((i + 1) / (i + 2))
    log_comp = np.log(compositions.values)
    ilr_coords = log_comp @ V.T
    return pd.DataFrame(
        ilr_coords,
        index=compositions.index,
        columns=[f"ilr_{j+1}" for j in range(D - 1)],
    )


# ── T3.3.3 — Verificación: isometría de Aitchison ────────────────────────────

def aitchison_distance(comp1: np.ndarray, comp2: np.ndarray) -> float:
    """
    Distancia de Aitchison entre dos composiciones (vectores 1-D).

    d_A(x, y) = ||clr(x) - clr(y)||₂  = ||clr(x/y)||₂
    """
    log_ratio = np.log(comp1 / comp2)
    clr_diff  = log_ratio - log_ratio.mean()
    return float(np.sqrt(np.sum(clr_diff ** 2)))


def verify_ilr_isometry(
    compositions: pd.DataFrame,
    ilr_coords: pd.DataFrame,
    n_pairs: int = 200,
    seed: int = 42,
) -> dict:
    """
    T3.3.3 — Verifica que d_Aitchison(wᵢ, wⱼ) ≈ d_Euclidean(ilr(wᵢ), ilr(wⱼ)).

    Muestrea n_pairs pares aleatorios y compara ambas distancias.
    Retorna diccionario con estadísticos del error relativo.
    """
    rng = np.random.default_rng(seed)
    N = len(compositions)
    idx_pairs = rng.integers(0, N, size=(n_pairs, 2))

    comp_vals = compositions.values
    ilr_vals  = ilr_coords.values
    errors    = []

    for i, j in idx_pairs:
        if i == j:
            continue
        d_a = aitchison_distance(comp_vals[i], comp_vals[j])
        d_e = float(np.linalg.norm(ilr_vals[i] - ilr_vals[j]))
        if d_a > 1e-12:
            errors.append(abs(d_a - d_e) / d_a)

    errors = np.array(errors)
    return {
        "n_pairs_tested": len(errors),
        "max_relative_error": float(errors.max()),
        "mean_relative_error": float(errors.mean()),
        "pct_below_1e-10": float((errors < 1e-10).mean() * 100),
        "passed": bool(errors.max() < 1e-8),
    }


# ── Figura T3.3 ───────────────────────────────────────────────────────────────

def _plot_logratio_summary(
    compositions: pd.DataFrame,
    clr_coords: pd.DataFrame,
    ilr_coords: pd.DataFrame,
    verification: dict,
) -> Path:
    """Cuatro paneles: distribuciones CLR, biplot ILR, varianza ILR, verificación."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # ── Panel 1: distribución de coordenadas CLR por mes ──
    clr_vals = [clr_coords[c].values for c in clr_coords.columns]
    bp = axes[0, 0].boxplot(
        clr_vals,
        tick_labels=MONTH_NAMES,
        patch_artist=True,
        medianprops={"color": "red", "linewidth": 1.5},
        flierprops={"marker": ".", "markersize": 2, "alpha": 0.3},
    )
    wet = {4, 5, 6, 7, 8, 9}          # índices 0-based en MONTH_NAMES: May-Oct = 4..9
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor("#FFCDD2" if i in wet else "#BBDEFB")
    axes[0, 0].axhline(0, color="grey", linestyle="--", linewidth=0.8)
    axes[0, 0].set_title("Distribución de coordenadas CLR por mes\n"
                          "(rojo=húmedo, azul=seco)", fontsize=10)
    axes[0, 0].set_ylabel("CLR")
    axes[0, 0].tick_params(axis="x", rotation=45)

    # ── Panel 2: biplot ILR1 vs ILR2 (secos vs húmedos / invierno vs prim.seca) ──
    axes[0, 1].scatter(
        ilr_coords["ilr_1"], ilr_coords["ilr_2"],
        s=8, alpha=0.4, color="#1565C0", edgecolors="none",
    )
    axes[0, 1].set_xlabel("ILR-1  (secos vs húmedos)", fontsize=9)
    axes[0, 1].set_ylabel("ILR-2  (invierno vs prim.seca)", fontsize=9)
    axes[0, 1].set_title(
        f"Biplot ILR1 vs ILR2\n(N={len(ilr_coords):,} estaciones)", fontsize=10
    )

    # ── Panel 3: varianza explicada por cada coordenada ILR ──
    variances = ilr_coords.var(axis=0).values
    total_var  = variances.sum()
    pct_var    = variances / total_var * 100
    cumul_var  = np.cumsum(pct_var)
    ax2        = axes[1, 0].twinx()
    axes[1, 0].bar(range(1, 12), pct_var, color="#1565C0", alpha=0.7,
                   label="% varianza")
    ax2.plot(range(1, 12), cumul_var, "o-", color="#C62828",
             linewidth=1.5, markersize=4, label="Acumulada")
    ax2.axhline(80, color="grey", linestyle=":", linewidth=1)
    ax2.set_ylabel("% acumulada", fontsize=9)
    axes[1, 0].set_xlabel("Coordenada ILR")
    axes[1, 0].set_ylabel("% varianza", fontsize=9)
    axes[1, 0].set_xticks(range(1, 12))
    axes[1, 0].set_title("Varianza por coordenada ILR\n"
                          "(indica jerarquía de la SBP)", fontsize=10)
    axes[1, 0].legend(loc="upper right", fontsize=8)
    ax2.legend(loc="center right", fontsize=8)

    # ── Panel 4: verificación de isometría d_A vs d_E ──
    rng = np.random.default_rng(42)
    N = len(compositions)
    idx_pairs = rng.integers(0, N, size=(500, 2))
    d_ait, d_euc = [], []
    for i, j in idx_pairs:
        if i == j:
            continue
        d_ait.append(aitchison_distance(
            compositions.values[i], compositions.values[j]
        ))
        d_euc.append(float(np.linalg.norm(
            ilr_coords.values[i] - ilr_coords.values[j]
        )))
    d_ait = np.array(d_ait)
    d_euc = np.array(d_euc)
    axes[1, 1].scatter(d_ait, d_euc, s=8, alpha=0.4, color="#2E7D32",
                       edgecolors="none")
    lim = max(d_ait.max(), d_euc.max()) * 1.05
    axes[1, 1].plot([0, lim], [0, lim], "r--", linewidth=1.2, label="y=x")
    axes[1, 1].set_xlabel("d_Aitchison(wᵢ, wⱼ)")
    axes[1, 1].set_ylabel("d_Euclidean(ilr(wᵢ), ilr(wⱼ))")
    axes[1, 1].set_title(
        f"Verificación de isometría\n"
        f"Max error rel.: {verification['max_relative_error']:.2e}  "
        f"{'✓ PASA' if verification['passed'] else '✗ FALLA'}",
        fontsize=10,
    )
    axes[1, 1].legend(fontsize=9)

    fig.suptitle("T3.3 — Transformaciones Log-Ratio (CLR e ILR)", fontsize=12)
    fig.tight_layout()

    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / "coda_logratio_transforms.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


# ── Punto de entrada T3.3 ─────────────────────────────────────────────────────

def run_t3_3(verbose: bool = True) -> pd.DataFrame:
    """
    Ejecuta T3.3 completo.

    Genera:
      - data/processed/composiciones_clr.parquet   (N × 12, Σclr=0)
      - data/processed/composiciones_ilr.parquet   (N × 11, SBP climatológica)
      - outputs/figures/coda_logratio_transforms.png

    Retorna el DataFrame de coordenadas ILR (usado en T3.4 para clustering).
    """
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("=" * 60)
        print("T3.3 — Transformaciones Log-Ratio")
        print("=" * 60)

    comp = pd.read_parquet(DATA_PROCESSED / "compositions_no_zeros.parquet")
    if verbose:
        print(f"\nComposiciones cargadas: {comp.shape[0]:,} estaciones × {comp.shape[1]} meses")
        print(f"Ceros: {(comp==0).values.sum()}  NaN: {comp.isna().values.sum()}")

    # ── T3.3.1 — CLR ─────────────────────────────────────────────────────────
    if verbose:
        print("\n[T3.3.1] CLR (Centered Log-Ratio)...")
    clr_coords = clr_transform(comp)
    clr_sum_check = clr_coords.sum(axis=1)
    if verbose:
        print(f"  Shape: {clr_coords.shape}")
        print(f"  Σ(clr) por fila: max|sum| = {clr_sum_check.abs().max():.2e}  (debe → 0)")
        print(f"  Rango CLR: [{clr_coords.values.min():.3f}, {clr_coords.values.max():.3f}]")

    # ── T3.3.2 — ILR con SBP climatológica ───────────────────────────────────
    if verbose:
        print("\n[T3.3.2] ILR con SBP climatológica (11 particiones)...")

    sbp  = build_sbp_matrix()
    psi  = sbp_to_contrast(sbp)

    gram_err = np.linalg.norm(psi @ psi.T - np.eye(11))
    if verbose:
        print(f"  SBP: {sbp.shape}  →  Ψ: {psi.shape}")
        print(f"  Ortonormalidad ||Ψ Ψᵀ − I||_F = {gram_err:.2e}  (debe → 0)")

    ilr_sbp = ilr_transform(comp, psi)

    if verbose:
        print(f"  Shape ILR-SBP: {ilr_sbp.shape}")
        variances = ilr_sbp.var(axis=0)
        cumul = variances.cumsum() / variances.sum() * 100
        n80 = int((cumul < 80).sum()) + 1
        print(f"  Varianza por coord (%):")
        for k in range(11):
            bar = "█" * int(variances.iloc[k]/variances.max()*20)
            print(f"    ILR-{k+1:2d}: {variances.iloc[k]:6.4f}  "
                  f"({variances.iloc[k]/variances.sum()*100:5.1f}%)  "
                  f"acum={cumul.iloc[k]:5.1f}%  {bar}")
        print(f"  Coordenadas para explicar ≥80% varianza: {n80}")

    # ILR Helmert (referencia)
    ilr_helm = helmert_ilr(comp)
    if verbose:
        print(f"\n  ILR-Helmert (referencia): {ilr_helm.shape}")

    # ── T3.3.3 — Verificación isometría ──────────────────────────────────────
    if verbose:
        print("\n[T3.3.3] Verificando isometría d_Aitchison ≡ d_Euclidean(ILR)...")
    verif = verify_ilr_isometry(comp, ilr_sbp, n_pairs=500)
    if verbose:
        print(f"  Pares testeados: {verif['n_pairs_tested']}")
        print(f"  Error relativo máximo: {verif['max_relative_error']:.2e}")
        print(f"  Error relativo medio:  {verif['mean_relative_error']:.2e}")
        status = "✓ PASA" if verif["passed"] else "✗ FALLA"
        print(f"  Resultado: {status}")

    # ── T3.3.4 — Guardar ─────────────────────────────────────────────────────
    clr_path = DATA_PROCESSED / "composiciones_clr.parquet"
    ilr_path = DATA_PROCESSED / "composiciones_ilr.parquet"
    clr_coords.to_parquet(clr_path)
    ilr_sbp.to_parquet(ilr_path)
    if verbose:
        print(f"\n[Guardado] {clr_path.name}  ({clr_coords.shape})")
        print(f"[Guardado] {ilr_path.name}  ({ilr_sbp.shape})")

    # ── Figura ────────────────────────────────────────────────────────────────
    fig_path = _plot_logratio_summary(comp, clr_coords, ilr_sbp, verif)
    if verbose:
        print(f"[Figura]  {fig_path.name}")

    # ── Resumen estadístico ───────────────────────────────────────────────────
    if verbose:
        print("\n── Estadísticos ILR-SBP ──")
        desc = ilr_sbp.describe().loc[["mean","std","min","max"]]
        for col in ilr_sbp.columns[:6]:
            row = desc[col]
            print(f"  {col}: media={row['mean']:+.3f}  std={row['std']:.3f}  "
                  f"[{row['min']:.2f}, {row['max']:.2f}]")
        print("  ...")

    print("\n[OK] T3.3 completado.")
    return ilr_sbp


if __name__ == "__main__":
    run_t3_3(verbose=True)
