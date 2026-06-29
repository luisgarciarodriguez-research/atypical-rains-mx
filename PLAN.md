# PLAN DE ACCIÓN — Análisis Pluviométrico México (2013–2026)

> **Proyecto**: Análisis exploratorio, detección de anomalías y clustering composicional de datos de precipitación de estaciones de monitoreo en México.
>
> **Dataset**: `data/raw/stats_lluvia_2013_2026_datos_flt.csv`
>
> **Última actualización**: 2026-06-29

---

## 0. Configuración del Proyecto

### 0.1 Estructura de directorios

```
atypical_rains_mx/
├── PLAN.md                          # ← este archivo
├── environment.yml                  # conda env specification
├── requirements.txt                 # pip fallback
├── data/
│   ├── raw/                         # datos originales (read-only)
│   │   └── stats_lluvia_2013_2026_datos_flt.csv
│   ├── processed/                   # datos limpios intermedios
│   │   ├── lluvia_clean.parquet     # T1.1 output
│   │   ├── lluvia_imputed.parquet   # post-imputación
│   │   └── composiciones_ilr.parquet # T3.3 output
│   └── catalogs/                    # catálogos de referencia
│       └── anomalias_catalogo.csv   # T2.consolidación output
├── notebooks/
│   ├── 01_eda.ipynb                 # Fase I completa
│   ├── 02_anomalias.ipynb           # Fase II completa
│   └── 03_coda_clustering.ipynb     # Fase III completa
├── run_all.sh                       # ejecuta el pipeline completo T1–T3.5
├── src/
│   ├── __init__.py
│   ├── config.py                    # constantes, paths, seeds
│   ├── loading.py                   # T1.1 — carga y limpieza inicial
│   ├── missing.py                   # T1.2 — diagnóstico de faltantes
│   ├── distributions.py             # T1.3 — análisis distributivo
│   ├── spatial.py                   # T1.4 — geoespacial + kriging
│   ├── anomalies.py                 # T2.1–T2.4 — capas de detección
│   ├── consolidation.py             # T2.5 — consenso multi-capa
│   ├── coda_prep.py                 # T3.1–T3.2 — selección y ceros CoDA
│   ├── compositional.py             # T3.3 — transformaciones log-ratio
│   ├── clustering.py                # T3.4 — K-Means, jerárquico, GMM
│   ├── validation.py                # T3.5 — validación e interpretación
│   ├── voronoi_map.py               # mapa de teselación de Voronoi (k=28)
│   ├── report.py                    # generación de reporte PDF
│   └── slides.py                    # generación de presentación PPTX
├── outputs/
│   ├── figures/                     # gráficas exportadas
│   └── reports/                     # reportes generados
└── tests/
    ├── test_loading.py
    ├── test_compositional.py
    └── test_anomalies.py
```

### 0.2 Dependencias

```bash
# Crear entorno
conda create -n lluvia python=3.11 -y
conda activate lluvia

# Core
pip install pandas numpy scipy scikit-learn statsmodels

# Geoespacial
pip install geopandas contextily pysal pykrige folium

# Visualización
pip install matplotlib seaborn plotly missingno

# Datos faltantes
pip install pyampute

# Deep learning (solo para T2.4.2)
pip install torch

# CoDA (Python experimental — para R se usa compositions/zCompositions)
pip install scikit-coda  # intentar; si falla, implementar manual

# Utilidades
pip install pyarrow openpyxl tqdm jupyter

# Verificar instalación
python -c "import pandas, numpy, scipy, sklearn, geopandas, pykrige, missingno; print('OK')"
```

### 0.3 Constantes del proyecto — `src/config.py`

```python
"""Constantes globales del proyecto."""
from pathlib import Path

# ── Paths ──
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_CATALOGS = ROOT / "data" / "catalogs"
FIGURES = ROOT / "outputs" / "figures"
REPORTS = ROOT / "outputs" / "reports"

RAW_FILE = DATA_RAW / "stats_lluvia_2013_2026_datos_flt.csv"

# ── Dataset constants ──
MISSING_CODE = -99.0
SEPARATOR = "\t"
META_COLS = ["#Station", "ID_st", "State", "Name", "No.records", "Long", "Lat"]
DROP_COLS = ["ID_st", "Unnamed: 168"]  # columnas a eliminar

# ── Temporal ──
YEAR_START = 2013
YEAR_END = 2026
MONTHS_LAST_YEAR = 5  # solo ene–may de 2026
TOTAL_MONTHS = (2025 - 2013 + 1) * 12 + MONTHS_LAST_YEAR  # = 161

# ── Meses húmedos y secos (referencia climatológica México) ──
WET_MONTHS = [5, 6, 7, 8, 9, 10]    # mayo–octubre
DRY_MONTHS = [11, 12, 1, 2, 3, 4]   # noviembre–abril

# ── Reproducibilidad ──
SEED = 42

# ── Geoespacial ──
MEXICO_BBOX = {
    "lon_min": -118.5, "lon_max": -86.5,
    "lat_min": 14.5,   "lat_max": 33.0,
}

# ── Umbrales de anomalías ──
ZSCORE_THRESHOLD = 3.0
PERCENTILE_LOW = 2
PERCENTILE_HIGH = 98
LOF_K_NEIGHBORS = 20
ISOLATION_FOREST_CONTAMINATION = "auto"
MAHAL_CHI2_ALPHA = 0.025  # para χ²(12, 0.975)

# ── Clustering ──
K_RANGE = range(2, 16)
BOOTSTRAP_JACCARD_THRESHOLD = 0.75
```

---

## Diagnóstico del Dataset (ya ejecutado)

Resultados del perfilado previo que condicionan todo el diseño:

| Métrica | Valor | Implicación |
|---|---|---|
| Estaciones | 1,959 | Cobertura nacional (32 estados) |
| Meses | 161 (ene-2013 → may-2026) | Serie de 13+ años |
| % faltantes (-99) | 46.2% | Condiciona toda la estrategia |
| Peor año | 2017 (77.2% faltante) | Subperíodo 2013–2017 poco confiable |
| Distribución | Sesgada derecha (μ=79.6, med=32.0, σ=116.3) | Descartar supuestos de normalidad |
| % ceros | 16.3% | Conflicto directo con log-ratios (CoDA) |
| Estaciones con 0 datos | 14 | Excluir directamente |
| Estaciones con 100% datos | 0 | Ninguna es completa |
| Máximo | 1,894 mm (EZAPATACFE, CHIS, jun-2017) | Evento extremo, no necesariamente anomalía |

---

## FASE I — Análisis Exploratorio de Datos

### T1.1 Limpieza y Preparación — `src/loading.py`

**Objetivo**: Producir un DataFrame limpio en `data/processed/lluvia_clean.parquet`.

- [ ] **T1.1.1** Cargar CSV con `sep='\t'`. Reemplazar todos los `-99.0` por `np.nan`. Eliminar columnas `ID_st` y `Unnamed: 168`.
- [ ] **T1.1.2** Validar tipos: `Long`/`Lat` → float64, `No.records` → int, columnas de precipitación → float64. Detectar y listar valores con precisión excesiva (>2 decimales; e.g., `104.1500015`).
- [ ] **T1.1.3** Validar coordenadas contra `MEXICO_BBOX`. Listar estaciones fuera del polígono. Generar scatter plot lon/lat con contorno de México (`geopandas` + Natural Earth shapefile).
- [ ] **T1.1.4** Verificar coherencia: comparar `No.records` vs conteo real de `notna()` por fila. Reportar discrepancias.
- [ ] **T1.1.5** Construir columnas auxiliares: `pct_complete` (% de meses con dato válido por estación), `years_active` (años con al menos 6 meses de datos).
- [ ] **T1.1.6** Guardar como parquet: `data/processed/lluvia_clean.parquet`.

```python
# Patrón para extraer año y mes de los nombres de columna
import re

def parse_rain_columns(columns):
    """Parsea '1_2013' → (mes=1, año=2013). Retorna dict {col_name: (month, year)}."""
    rain_cols = {}
    for col in columns:
        match = re.match(r'^(\d{1,2})_(\d{4})$', col)
        if match:
            rain_cols[col] = (int(match.group(1)), int(match.group(2)))
    return rain_cols
```

**Criterio de aceptación**: DataFrame sin valores centinela, tipos correctos, parquet generado sin errores. `assert df[rain_cols].min().min() >= 0 or np.isnan(...)`.

---

### T1.2 Diagnóstico de Datos Faltantes — `src/missing.py`

**Objetivo**: Clasificar el mecanismo de datos faltantes (MCAR / MAR / MNAR).

- [ ] **T1.2.1** Generar heatmap de datos faltantes (estaciones × meses) con `missingno.matrix()`. Ordenar estaciones por estado y por `pct_complete`.

```python
import missingno as msno
# Ordenar por estado + completitud para revelar patrones de bloque
df_sorted = df.sort_values(["State", "pct_complete"], ascending=[True, False])
msno.matrix(df_sorted[rain_cols], figsize=(20, 12), sparkline=False)
```

- [ ] **T1.2.2** Test de Little para MCAR. Si el paquete `pyampute` no incluye Little's test, implementar manualmente:

```python
# Little's MCAR test simplificado
# H0: datos son MCAR
# Si p < 0.05 → rechazar MCAR → investigar MAR vs MNAR
from scipy.stats import chi2

def littles_mcar_test(data):
    """
    Implementación simplificada del test de Little (1988).
    Agrupa patrones de datos faltantes, calcula estadístico χ² basado
    en desviaciones de medias de subgrupos vs media global.
    """
    # Identificar patrones únicos de datos faltantes
    patterns = data.notna().astype(int)
    pattern_ids = patterns.apply(lambda row: tuple(row), axis=1)
    unique_patterns = pattern_ids.unique()

    grand_mean = data.mean()
    grand_cov = data.cov()

    chi2_stat = 0
    for pattern in unique_patterns:
        mask = pattern_ids == pattern
        subgroup = data[mask]
        n_j = len(subgroup)
        if n_j < 2:
            continue
        observed_cols = [c for c, v in zip(data.columns, pattern) if v == 1]
        if len(observed_cols) == 0:
            continue
        sub_mean = subgroup[observed_cols].mean()
        diff = sub_mean - grand_mean[observed_cols]
        sub_cov = grand_cov.loc[observed_cols, observed_cols] / n_j
        try:
            chi2_stat += float(diff @ np.linalg.pinv(sub_cov) @ diff)
        except np.linalg.LinAlgError:
            continue

    df_chi = sum(len([v for v in p if v == 1]) for p in unique_patterns) - len(data.columns)
    p_value = 1 - chi2.cdf(chi2_stat, df_chi)
    return chi2_stat, df_chi, p_value
```

- [ ] **T1.2.3** Correlación de datos faltantes: calcular correlación punto-biserial entre indicador de ausencia (1=faltante) y variables observadas (lat, lon, mes, año).

```python
from scipy.stats import pointbiserialr

# Para cada columna de lluvia, correlacionar su datos faltantes con lat/lon
missing_indicator = df[rain_cols].isna().astype(int)
for meta_var in ["Lat", "Long"]:
    correlations = {}
    for col in rain_cols:
        r, p = pointbiserialr(missing_indicator[col], df[meta_var])
        correlations[col] = (r, p)
    # Resumir: ¿los faltantes correlacionan con la ubicación?
```

- [ ] **T1.2.4** Segmentación temporal de dropout: identificar bloques contiguos de NaN por estación. Clasificar patrón como monotónico (la estación deja de reportar y no regresa) vs intermitente.

**Criterio de aceptación**: Reporte textual con clasificación MCAR/MAR/MNAR, visualizaciones guardadas en `outputs/figures/missing_*.png`.

**Entregable**: `outputs/reports/diagnostico_datos_faltantes.md`

---

### T1.3 Caracterización Distribucional — `src/distributions.py`

- [ ] **T1.3.1** Histograma + KDE de precipitación mensual en escala original y log(x+1). Calcular skewness y kurtosis con `scipy.stats.skew` / `kurtosis`.

```python
from scipy.stats import skew, kurtosis
rain_valid = df[rain_cols].values.flatten()
rain_valid = rain_valid[~np.isnan(rain_valid)]
print(f"Skewness: {skew(rain_valid):.3f}")
print(f"Kurtosis: {kurtosis(rain_valid):.3f}")  # excess kurtosis
# Log-transform (agregar 1 para manejar ceros)
rain_log = np.log1p(rain_valid)
```

- [ ] **T1.3.2** Boxplot estacional: agrupar todos los valores por mes (1–12) ignorando año y estación. Calcular PCI (Precipitation Concentration Index) por estación:

```python
def pci(monthly_totals):
    """
    PCI = (Σ pᵢ²) / (Σ pᵢ)² × 100
    PCI ≈ 8.3 → uniforme; PCI > 20 → fuerte concentración estacional
    Oliver (1980).
    """
    p = np.array(monthly_totals)
    p = p[~np.isnan(p)]
    if len(p) < 12 or p.sum() == 0:
        return np.nan
    return (np.sum(p**2) / (np.sum(p)**2)) * 100 * 12  # ×12 por 12 meses
```

- [ ] **T1.3.3** Heatmap 32 estados × 12 meses: mediana de precipitación. Agregar dendrograma lateral con `sns.clustermap`.
- [ ] **T1.3.4** Descomposición STL de la serie nacional (mediana mensual de todas las estaciones). Test de Mann-Kendall para tendencia.

```python
from statsmodels.tsa.seasonal import STL
# Serie temporal: mediana nacional mensual
national_median = df[rain_cols].median(axis=0)
# Convertir a serie con índice temporal
import pandas as pd
dates = pd.date_range("2013-01", periods=161, freq="MS")
ts = pd.Series(national_median.values, index=dates)
stl = STL(ts, period=12, robust=True)
result = stl.fit()
result.plot()
```

- [ ] **T1.3.5** Semivariograma empírico de la precipitación anual media (restringir a estaciones con ≥80% completitud). Ajustar modelo esférico o exponencial.

```python
from pykrige.ok import OrdinaryKriging
# Precipitación anual media por estación
annual_mean = df[rain_cols].mean(axis=1)
# Filtrar estaciones con buena cobertura
mask_complete = df["pct_complete"] >= 0.80
OK = OrdinaryKriging(
    df.loc[mask_complete, "Long"].values,
    df.loc[mask_complete, "Lat"].values,
    annual_mean[mask_complete].values,
    variogram_model="spherical",
    verbose=False,
)
# OK.get_variogram_points() para graficar
```

**Entregable**: Figuras en `outputs/figures/dist_*.png`, tabla de momentos en `outputs/reports/distribution_summary.csv`.

---

### T1.4 Visualización Geoespacial — `src/spatial.py`

> **Dependencia de datos**: `data/raw/ne_50m_admin_0_countries.zip`
> Shapefile Natural Earth Admin 0 Countries, resolución 50 m (v5.x). Se usa
> para trazar el polígono de México (T1.4.1–T1.4.3) y enmascarar los puntos de
> kriging fuera del territorio nacional (T1.4.2). La resolución 50 m es
> suficiente para este propósito y es significativamente más ligera que la
> versión 10 m. El archivo está incluido en el repositorio; si es necesario
> volver a descargarlo:
> `https://naciscdn.org/naturalearth/50m/cultural/ne_50m_admin_0_countries.zip`

- [ ] **T1.4.1** Mapa de cobertura: scatter de 1,959 estaciones coloreado por `pct_complete`, con contorno de México y topografía.

```python
import geopandas as gpd
import contextily as cx

mexico = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
mexico = mexico[mexico.name == "Mexico"]

gdf = gpd.GeoDataFrame(
    df, geometry=gpd.points_from_xy(df["Long"], df["Lat"]), crs="EPSG:4326"
)
ax = mexico.to_crs(epsg=3857).plot(figsize=(14, 10), alpha=0.1, edgecolor="black")
gdf_web = gdf.to_crs(epsg=3857)
gdf_web.plot(
    ax=ax, column="pct_complete", cmap="RdYlGn", markersize=8,
    legend=True, legend_kwds={"label": "% Completitud"}
)
cx.add_basemap(ax, source=cx.providers.CartoDB.Positron)
```

- [ ] **T1.4.2** Mapa de precipitación anual media interpolada (kriging ordinario con el modelo de T1.3.5).
- [ ] **T1.4.3** Cuatro mapas trimestrales (DEF, MAM, JJA, SON) con la mediana de precipitación por estación.

**Entregable**: Figuras en `outputs/figures/map_*.png`.

---

## FASE II — Análisis de Anomalías

> **Precondición**: Fase I completada. `data/processed/lluvia_clean.parquet` disponible.

### T2.1 Capa 1 — Artefactos Instrumentales — `src/anomalies.py`

- [ ] **T2.1.1** Ceros sospechosos: flag = cero en mes húmedo (may–oct) cuando la mediana regional (mismo estado, mismo mes) > 50 mm.

```python
def flag_suspicious_zeros(df, rain_cols, rain_col_map):
    """
    Identifica ceros potencialmente instrumentales.
    rain_col_map: dict {col_name: (month, year)}
    """
    flags = pd.DataFrame(False, index=df.index, columns=rain_cols)
    for col, (month, year) in rain_col_map.items():
        if month not in WET_MONTHS:
            continue
        for state in df["State"].unique():
            state_mask = df["State"] == state
            regional_median = df.loc[state_mask, col].median()
            if pd.isna(regional_median) or regional_median <= 50:
                continue
            zero_mask = state_mask & (df[col] == 0.0)
            flags.loc[zero_mask, col] = True
    return flags
```

- [ ] **T2.1.2** Valores constantes: detectar ≥3 meses consecutivos con el mismo valor no-cero por estación.

```python
def flag_stuck_sensor(df, rain_cols, min_consecutive=3):
    """Detecta secuencias de valores idénticos (posible sensor atascado)."""
    flags = pd.DataFrame(False, index=df.index, columns=rain_cols)
    for idx in df.index:
        vals = df.loc[idx, rain_cols].values
        count = 1
        for j in range(1, len(vals)):
            if vals[j] == vals[j-1] and not np.isnan(vals[j]) and vals[j] != 0:
                count += 1
                if count >= min_consecutive:
                    for k in range(j - count + 1, j + 1):
                        flags.iloc[idx, k] = True
            else:
                count = 1
    return flags
```

- [ ] **T2.1.3** Verificar ausencia de valores negativos residuales (distintos de -99, que ya se recodificaron).
- [ ] **T2.1.4** Detectar valores con precisión anómala (>2 decimales).

```python
def flag_precision_anomalies(df, rain_cols, max_decimals=2):
    """Valores como 104.1500015 sugieren errores de punto flotante."""
    flags = pd.DataFrame(False, index=df.index, columns=rain_cols)
    for col in rain_cols:
        vals = df[col].dropna()
        # Contar decimales significativos
        remainder = vals - vals.round(max_decimals)
        anomalous = remainder.abs() > 1e-9
        flags.loc[anomalous.index[anomalous], col] = True
    return flags
```

**Entregable**: `data/catalogs/flags_capa1.parquet` — DataFrame booleano (estaciones × meses).

---

### T2.2 Capa 2 — Anomalías Univariadas Contextualizadas

- [ ] **T2.2.1** Z-score estacional por estación usando mediana y MAD (Median Absolute Deviation) en lugar de media y σ:

```python
from scipy.stats import median_abs_deviation

def zscore_seasonal(df, rain_cols, rain_col_map, threshold=3.0):
    """
    Z-score robusto por estación-mes.
    z = (x - median_mes) / MAD_mes
    """
    flags = pd.DataFrame(False, index=df.index, columns=rain_cols)
    scores = pd.DataFrame(np.nan, index=df.index, columns=rain_cols)

    for idx in df.index:
        # Agrupar las columnas por mes (1–12)
        for month in range(1, 13):
            month_cols = [c for c, (m, y) in rain_col_map.items() if m == month]
            vals = df.loc[idx, month_cols].dropna().values
            if len(vals) < 3:
                continue
            med = np.median(vals)
            mad = median_abs_deviation(vals, scale="normal")
            if mad == 0:
                continue
            for col in month_cols:
                v = df.loc[idx, col]
                if np.isnan(v):
                    continue
                z = (v - med) / mad
                scores.loc[idx, col] = z
                if abs(z) > threshold:
                    flags.loc[idx, col] = True
    return flags, scores
```

- [ ] **T2.2.2** Percentiles condicionales: flag si valor < P2 o > P98 del historial estación-mes.
- [ ] **T2.2.3** Adjusted boxplot (Hubert & Vandervieren, 2008) por cluster regional. Usar `medcouple` de `statsmodels.stats.stattools`.

```python
from statsmodels.stats.stattools import medcouple_1d

def adjusted_boxplot_fences(data):
    """Fences del boxplot ajustado para distribuciones sesgadas."""
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    mc = medcouple_1d(data)
    if mc >= 0:
        lower = q1 - 1.5 * np.exp(-4 * mc) * iqr
        upper = q3 + 1.5 * np.exp(3 * mc) * iqr
    else:
        lower = q1 - 1.5 * np.exp(-3 * mc) * iqr
        upper = q3 + 1.5 * np.exp(4 * mc) * iqr
    return lower, upper
```

**Entregable**: `data/catalogs/flags_capa2.parquet`, `data/catalogs/zscores.parquet`.

---

### T2.3 Capa 3 — Anomalías Espaciales

- [ ] **T2.3.1** Residuos de kriging: para cada mes con cobertura suficiente, ajustar kriging ordinario y flag estaciones con |residuo| > 2.5σ.

```python
from pykrige.ok import OrdinaryKriging

def kriging_residual_flags(df, rain_cols, rain_col_map, threshold_sigma=2.5, min_stations=50):
    """Anomalías espaciales vía residuos de kriging."""
    flags = pd.DataFrame(False, index=df.index, columns=rain_cols)
    for col in rain_cols:
        valid_mask = df[col].notna()
        if valid_mask.sum() < min_stations:
            continue
        lons = df.loc[valid_mask, "Long"].values
        lats = df.loc[valid_mask, "Lat"].values
        vals = df.loc[valid_mask, col].values
        try:
            OK = OrdinaryKriging(lons, lats, vals, variogram_model="spherical", verbose=False)
            predicted, var = OK.execute("points", lons, lats)
            residuals = vals - predicted.flatten()
            sigma = np.std(residuals)
            anomalous = np.abs(residuals) > threshold_sigma * sigma
            valid_indices = df.index[valid_mask]
            flags.loc[valid_indices[anomalous], col] = True
        except Exception:
            continue
    return flags
```

- [ ] **T2.3.2** LOF (Local Outlier Factor) en espacio (lat, lon, precipitación) por mes:

```python
from sklearn.neighbors import LocalOutlierFactor

def lof_spatial_flags(df, col, k=20):
    """LOF en espacio conjunto (lat, lon, valor) para un mes dado."""
    valid = df[["Lat", "Long", col]].dropna()
    if len(valid) < k + 1:
        return pd.Series(False, index=df.index)
    # Normalizar las tres dimensiones
    from sklearn.preprocessing import StandardScaler
    X = StandardScaler().fit_transform(valid.values)
    lof = LocalOutlierFactor(n_neighbors=k, contamination="auto")
    labels = lof.fit_predict(X)  # -1 = outlier
    flags = pd.Series(False, index=df.index)
    flags.loc[valid.index] = (labels == -1)
    return flags
```

- [ ] **T2.3.3** LISA (Local Moran's I) con PySAL:

```python
from libpysal.weights import KNN
from esda.moran import Moran_Local

def lisa_flags(df, col, k=8, significance=0.05):
    """Identificar estaciones High-Low o Low-High (anomalías espaciales)."""
    valid = df[["Lat", "Long", col]].dropna()
    if len(valid) < k + 1:
        return pd.Series(False, index=df.index)
    coords = valid[["Long", "Lat"]].values
    w = KNN.from_array(coords, k=k)
    w.transform = "R"
    lisa = Moran_Local(valid[col].values, w, permutations=999)
    # Quadrants: 1=HH, 2=LH, 3=LL, 4=HL
    # Anomalías = LH (2) o HL (4) con p < significance
    spatial_anomaly = ((lisa.q == 2) | (lisa.q == 4)) & (lisa.p_sim < significance)
    flags = pd.Series(False, index=df.index)
    flags.loc[valid.index] = spatial_anomaly
    return flags
```

**Entregable**: `data/catalogs/flags_capa3.parquet`.

---

### T2.4 Capa 4 — Anomalías Multivariadas (Perfil Anual)

> **Nota**: Operar sobre vectores de 12 componentes (un perfil anual por estación-año). Requiere imputación previa de faltantes dentro de cada año o restricción a estaciones-año completas.

- [ ] **T2.4.1** Isolation Forest sobre perfiles anuales:

```python
from sklearn.ensemble import IsolationForest

def isolation_forest_profiles(profiles_df, contamination="auto", seed=42):
    """
    profiles_df: DataFrame con columnas mes_1...mes_12, filas = estación-año.
    Retorna boolean mask de anomalías.
    """
    clf = IsolationForest(contamination=contamination, random_state=seed, n_estimators=300)
    labels = clf.fit_predict(profiles_df.values)
    return labels == -1
```

- [ ] **T2.4.2** Autoencoder: arquitectura 12 → 6 → 3 → 6 → 12, ReLU, MSE loss. Umbral = percentil 95 del error de reconstrucción:

```python
import torch
import torch.nn as nn

class RainfallAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(12, 6), nn.ReLU(),
            nn.Linear(6, 3), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 6), nn.ReLU(),
            nn.Linear(6, 12),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

# Entrenamiento: usar log1p(perfil) como input para estabilizar magnitudes.
# Anomalía = MSE(x, x̂) > np.percentile(all_mse, 95)
```

- [ ] **T2.4.3** Distancia de Mahalanobis robusta (MCD) por cluster regional:

```python
from sklearn.covariance import MinCovDet
from scipy.stats import chi2

def mahalanobis_flags(profiles, alpha=0.025):
    """Mahalanobis robusta con MCD. Umbral: χ²(12, 1-α)."""
    mcd = MinCovDet(random_state=42).fit(profiles)
    distances = mcd.mahalanobis(profiles)
    threshold = chi2.ppf(1 - alpha, df=profiles.shape[1])
    return distances > threshold
```

**Entregable**: `data/catalogs/flags_capa4.parquet`.

---

### T2.5 Consolidación del Catálogo de Anomalías

- [ ] **T2.5.1** Calcular consenso multi-capa: anomalía confirmada = flag en ≥2 capas.

```python
def consolidate_flags(*flag_dfs):
    """Consenso multi-método. Retorna conteo de capas y clasificación."""
    stacked = sum(f.astype(int) for f in flag_dfs)
    confirmed = stacked >= 2
    return confirmed, stacked
```

- [ ] **T2.5.2** Clasificar cada anomalía confirmada:
  - **Artefacto instrumental**: predominantemente flaggeada por Capa 1 + otra capa.
  - **Evento extremo legítimo**: flaggeada por Capa 2 pero NO por Capa 3 (consistente con sus vecinos).
  - **Inconsistencia espacial**: flaggeada por Capa 3 (discrepante con vecinos).

- [ ] **T2.5.3** Calcular κ de Fleiss para concordancia inter-capa.

- [ ] **T2.5.4** Guardar catálogo final:

```python
# Esquema del catálogo
catalog_schema = {
    "station": str,        # #Station
    "state": str,          # State
    "month_col": str,      # e.g., '9_2020'
    "value_mm": float,     # valor observado
    "n_layers_flagged": int,  # cuántas capas lo detectaron
    "layer_1": bool,       # Capa 1 (instrumental)
    "layer_2": bool,       # Capa 2 (univariada)
    "layer_3": bool,       # Capa 3 (espacial)
    "layer_4": bool,       # Capa 4 (multivariada)
    "classification": str,  # artefacto | evento_extremo | inconsistencia_espacial
    "action": str,          # recodificar_nan | conservar_flag | investigar
}
# Guardar en data/catalogs/anomalias_catalogo.csv
```

**Entregable**: `data/catalogs/anomalias_catalogo.csv`.

---

## FASE III — Análisis de Cúmulos para Datos Composicionales

> **Precondición**: Fases I y II completadas. Catálogo de anomalías aplicado (artefactos → NaN).

### T3.1 Selección del Subconjunto Analítico

- [ ] **T3.1.1** Aplicar filtros secuenciales:

```python
def select_coda_subset(df, rain_col_map, min_months_per_year=10, min_years=3):
    """
    Filtro de inclusión para análisis composicional.
    Requiere al menos min_months_per_year con dato en al menos min_years años.
    """
    years = sorted(set(y for _, (m, y) in rain_col_map.items()))
    station_years = {}
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
```

- [ ] **T3.1.2** Excluir las 14 estaciones con 0 datos.
- [ ] **T3.1.3** Excluir estaciones con coordenadas erróneas (de T1.1.3).
- [ ] **T3.1.4** Recodificar artefactos instrumentales como NaN (del catálogo de anomalías).
- [ ] **T3.1.5** Reportar tamaño del subconjunto resultante (estimado: ~700–800 estaciones).

---

### T3.2 Tratamiento de Ceros

- [ ] **T3.2.1** Calcular composiciones brutas: para cada estación, promediar la precipitación mensual a través de los años disponibles, luego normalizar a proporciones (Σ=1).

```python
def compute_compositions(df, rain_col_map):
    """
    Calcular la composición media mensual (12 partes) por estación.
    Opción A: promediar primero, componer después.
    """
    monthly_means = pd.DataFrame(index=df.index, columns=range(1, 13), dtype=float)
    for month in range(1, 13):
        month_cols = [c for c, (m, y) in rain_col_map.items() if m == month]
        monthly_means[month] = df[month_cols].mean(axis=1)  # media a lo largo de años

    # Normalizar a composición (proporciones)
    row_sums = monthly_means.sum(axis=1)
    compositions = monthly_means.div(row_sums, axis=0)
    return compositions
```

- [ ] **T3.2.2** Contar y ubicar ceros en las composiciones. Un cero en la composición significa que la media mensual fue 0, es decir, nunca llovió en ese mes en todos los años registrados.
- [ ] **T3.2.3** Aplicar reemplazo multiplicativo como método base:

```python
def multiplicative_replacement(comp, delta_factor=0.65):
    """
    Reemplazo multiplicativo de ceros (Martín-Fernández et al., 2003).
    delta = delta_factor × min(valores no-cero).
    """
    result = comp.copy()
    for idx in result.index:
        row = result.loc[idx].values.astype(float)
        zeros = row == 0
        if not zeros.any():
            continue
        non_zero_min = row[~zeros].min()
        delta = delta_factor * non_zero_min
        n_zeros = zeros.sum()
        # Sustituir ceros
        row[zeros] = delta
        # Ajustar no-ceros para mantener Σ=1
        row[~zeros] = row[~zeros] * (1 - n_zeros * delta) / row[~zeros].sum()
        result.loc[idx] = row
    return result
```

- [ ] **T3.2.4** (Opcional / R) Aplicar lrEM con `zCompositions::lrEM()` como método principal. Interface Python→R via `rpy2` o script R standalone:

```r
# Script R: cero_treatment.R
library(zCompositions)
comp <- read.csv("data/processed/compositions_raw.csv", row.names=1)
comp_imputed <- lrEM(comp, label=0, dl=rep(0.001, ncol(comp)))
write.csv(comp_imputed, "data/processed/compositions_lrEM.csv")
```

- [ ] **T3.2.5** Análisis de sensibilidad: comparar composiciones resultantes de ambos métodos.

**Entregable**: `data/processed/compositions_no_zeros.parquet` (listo para transformación).

---

### T3.3 Transformaciones Log-Ratio — `src/compositional.py`

- [ ] **T3.3.1** Implementar CLR (Centered Log-Ratio):

```python
from scipy.stats import gmean

def clr_transform(compositions):
    """
    CLR: y_j = ln(w_j / g(w))
    donde g(w) = media geométrica de todas las partes.
    Retorna DataFrame con 12 columnas (pero rango 11 por la restricción Σ=0).
    """
    gm = gmean(compositions, axis=1)
    clr = np.log(compositions.div(gm, axis=0))
    return clr
```

- [ ] **T3.3.2** Implementar ILR (Isometric Log-Ratio) con SBP climatológica:

```python
import numpy as np

def build_sbp_matrix():
    """
    Sequential Binary Partition climatológicamente interpretable.
    Partición 1: meses secos (NDEFMA = 11,12,1,2,3,4) vs húmedos (MJJASO = 5,6,7,8,9,10)
    Particiones 2+: subdivisiones dentro de cada grupo.

    Retorna contrast matrix Ψ (11 × 12) para ILR.
    """
    # Partición definida como {+1, -1, 0}
    # Cada fila es una partición binaria
    # Columnas = meses 1–12
    partitions = [
        # P1: secos (11,12,1,2,3,4) = +1 vs húmedos (5,6,7,8,9,10) = -1
        [+1, +1, +1, +1, -1, -1, -1, -1, -1, -1, +1, +1],
        # P2: dentro de secos → invierno (12,1,2) vs primavera seca (3,4,11)
        [+1, +1, 0, 0, 0, 0, 0, 0, 0, 0, -1, +1],
        # P3: dentro de húmedos → inicio (5,6) vs pleno (7,8,9,10)
        [0, 0, 0, 0, +1, +1, -1, -1, -1, -1, 0, 0],
        # ... completar hasta 11 particiones para D=12
        # Las restantes particiones subdividen sucesivamente
    ]
    # NOTA: completar las 11 particiones o usar la Helmert sub-composition default.
    # Para una SBP completa, usar la función de composiciones (R) o implementar Helmert.
    pass  # TODO: completar con las 11 particiones

def ilr_transform(compositions, contrast_matrix):
    """
    ILR: y = ln(compositions) @ contrast_matrix.T
    contrast_matrix: (D-1) × D, ortonormal en el símplex.
    """
    log_comp = np.log(compositions)
    return log_comp @ contrast_matrix.T

def helmert_ilr(compositions):
    """
    ILR con base de Helmert (default cuando no se especifica SBP).
    Menos interpretable pero matemáticamente correcto.
    """
    D = compositions.shape[1]
    # Helmert sub-composition matrix
    V = np.zeros((D - 1, D))
    for i in range(D - 1):
        V[i, :i+1] = 1.0 / (i + 1)
        V[i, i+1] = -1.0
        V[i] *= np.sqrt((i + 1) / (i + 2))
    log_comp = np.log(compositions.values)
    ilr_coords = log_comp @ V.T
    return pd.DataFrame(ilr_coords, index=compositions.index,
                        columns=[f"ilr_{j+1}" for j in range(D-1)])
```

- [ ] **T3.3.3** Verificación: la distancia euclidiana entre coordenadas ILR debe coincidir con la distancia de Aitchison entre las composiciones originales.

```python
def aitchison_distance(comp1, comp2):
    """Distancia de Aitchison entre dos composiciones."""
    log_ratio = np.log(comp1 / comp2)
    D = len(comp1)
    clr1 = log_ratio - log_ratio.mean()
    return np.sqrt(np.sum(clr1**2))

# Verificar: d_Aitchison(w1, w2) == d_Euclidean(ilr(w1), ilr(w2))
```

- [ ] **T3.3.4** Guardar coordenadas ILR: `data/processed/composiciones_ilr.parquet`.

**Entregable**: `data/processed/composiciones_ilr.parquet` (11 columnas × N estaciones).

---

### T3.4 Clustering — `src/clustering.py`

- [ ] **T3.4.1** K-Means en espacio ILR:

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score

def kmeans_sweep(ilr_data, k_range=range(2, 16), seed=42):
    """Barrido de K con métricas de validación."""
    results = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=seed, n_init=20)
        labels = km.fit_predict(ilr_data)
        sil = silhouette_score(ilr_data, labels)
        ch = calinski_harabasz_score(ilr_data, labels)
        inertia = km.inertia_
        results.append({"k": k, "silhouette": sil, "calinski_harabasz": ch, "inertia": inertia})
    return pd.DataFrame(results)
```

- [ ] **T3.4.2** Gap statistic:

```python
def gap_statistic(data, k_range, n_bootstrap=50, seed=42):
    """Gap statistic (Tibshirani et al., 2001)."""
    rng = np.random.default_rng(seed)
    results = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        km.fit(data)
        log_Wk = np.log(km.inertia_)

        # Referencia uniforme
        log_Wk_refs = []
        for _ in range(n_bootstrap):
            ref_data = rng.uniform(data.min(axis=0), data.max(axis=0), size=data.shape)
            km_ref = KMeans(n_clusters=k, random_state=seed, n_init=5)
            km_ref.fit(ref_data)
            log_Wk_refs.append(np.log(km_ref.inertia_))

        gap = np.mean(log_Wk_refs) - log_Wk
        se = np.std(log_Wk_refs) * np.sqrt(1 + 1 / n_bootstrap)
        results.append({"k": k, "gap": gap, "se": se})
    return pd.DataFrame(results)
```

- [ ] **T3.4.3** Bootstrap de Jaccard para estabilidad (Hennig, 2007):

```python
def jaccard_bootstrap(data, k, n_bootstrap=100, seed=42):
    """Evalúa estabilidad de clusters via Jaccard bootstrap."""
    rng = np.random.default_rng(seed)
    km_full = KMeans(n_clusters=k, random_state=seed, n_init=20).fit(data)
    full_labels = km_full.labels_

    stabilities = []
    for _ in range(n_bootstrap):
        idx = rng.choice(len(data), size=len(data), replace=True)
        km_boot = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(data[idx])
        # Calcular Jaccard de cada cluster original vs mejor match en bootstrap
        # ... (implementación completa con matching húngaro)
    return np.mean(stabilities)  # > 0.75 = estable
```

- [ ] **T3.4.4** Clustering jerárquico con distancia de Aitchison (= euclidiana en ILR):

```python
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist

def hierarchical_clustering(ilr_data, method="ward"):
    """Clustering aglomerativo en espacio ILR."""
    dist_matrix = pdist(ilr_data, metric="euclidean")  # = Aitchison en ILR
    Z = linkage(dist_matrix, method=method)
    return Z

# Dendrograma
# dendrogram(Z, truncate_mode="lastp", p=30)
# Cortar: labels = fcluster(Z, t=k, criterion="maxclust")
```

- [ ] **T3.4.5** GMM (Gaussian Mixture Model) en espacio ILR:

```python
from sklearn.mixture import GaussianMixture

def gmm_sweep(ilr_data, k_range=range(2, 16), seed=42):
    """Barrido de componentes GMM con BIC."""
    results = []
    for k in k_range:
        gmm = GaussianMixture(n_components=k, random_state=seed, n_init=5, max_iter=300)
        gmm.fit(ilr_data)
        results.append({
            "k": k,
            "bic": gmm.bic(ilr_data),
            "aic": gmm.aic(ilr_data),
            "log_likelihood": gmm.score(ilr_data) * len(ilr_data),
        })
    return pd.DataFrame(results)
```

**Entregable**: Asignaciones de clusters por los tres métodos. `data/processed/cluster_assignments.parquet`.

---

### T3.5 Validación e Interpretación

- [ ] **T3.5.1** Concordancia inter-método:

```python
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

def method_concordance(labels_dict):
    """
    labels_dict: {"kmeans": array, "hierarchical": array, "gmm": array}
    Retorna matriz de ARI y NMI.
    """
    methods = list(labels_dict.keys())
    n = len(methods)
    ari_matrix = pd.DataFrame(np.eye(n), index=methods, columns=methods)
    nmi_matrix = pd.DataFrame(np.eye(n), index=methods, columns=methods)
    for i in range(n):
        for j in range(i+1, n):
            ari = adjusted_rand_score(labels_dict[methods[i]], labels_dict[methods[j]])
            nmi = normalized_mutual_info_score(labels_dict[methods[i]], labels_dict[methods[j]])
            ari_matrix.iloc[i, j] = ari_matrix.iloc[j, i] = ari
            nmi_matrix.iloc[i, j] = nmi_matrix.iloc[j, i] = nmi
    return ari_matrix, nmi_matrix
```

- [ ] **T3.5.2** Mapa de regímenes pluviométricos: proyectar clusters sobre mapa de México.

```python
def plot_regime_map(df, labels, title="Regímenes Pluviométricos"):
    """Mapa de México coloreado por cluster."""
    gdf = gpd.GeoDataFrame(
        df.assign(cluster=labels),
        geometry=gpd.points_from_xy(df["Long"], df["Lat"]),
        crs="EPSG:4326"
    )
    fig, ax = plt.subplots(figsize=(14, 10))
    mexico.plot(ax=ax, color="white", edgecolor="black", alpha=0.3)
    gdf.plot(ax=ax, column="cluster", cmap="Set2", markersize=12,
             legend=True, categorical=True)
    ax.set_title(title)
    plt.tight_layout()
    return fig
```

- [ ] **T3.5.3** Perfiles composicionales por cluster: barplot de 12 meses con media ± IC por cluster.
- [ ] **T3.5.4** Etiquetas climatológicas: asignar nombre interpretativo a cada cluster.
- [ ] **T3.5.5** Contraste con Köppen-Geiger y regiones hidrológicas CONAGUA.
- [ ] **T3.5.6** Estabilidad temporal (si se ejecutó Opción B año-por-año): fracción de años en cluster modal por estación.

**Entregable final**: `outputs/reports/regimenes_pluviometricos.md` con mapas, perfiles y tablas de concordancia.

---

## Orden de Ejecución y Dependencias

```
T1.1 ──→ T1.2 ──→ T1.3 ──→ T1.4
  │                  │
  │                  └──→ T1.3.5 (semivariograma) ──→ T2.3.1 (kriging residuals)
  │
  └──→ T2.1 ──→ T2.2 ──→ T2.3 ──→ T2.4 ──→ T2.5
                                                │
                                                └──→ T3.1 ──→ T3.2 ──→ T3.3 ──→ T3.4 ──→ T3.5
```

**Regla**: no iniciar una fase sin que la anterior esté completada y sus entregables verificados.

---

## Verificación Rápida por Tarea

```bash
# Verificar que los outputs existen y no están vacíos
python -c "
import pandas as pd
from pathlib import Path

checks = {
    'T1.1': 'data/processed/lluvia_clean.parquet',
    'T2.5': 'data/catalogs/anomalias_catalogo.csv',
    'T3.3': 'data/processed/composiciones_ilr.parquet',
    'T3.4': 'data/processed/cluster_assignments.parquet',
}
for task, path in checks.items():
    p = Path(path)
    status = '✓' if p.exists() and p.stat().st_size > 0 else '✗'
    print(f'  {status} {task}: {path}')
"
```

---

## Solicitudes Adicionales (actualizaciones post-implementación)

### 2026-06-29 — Preparación para publicación en revista científica

#### A. Traducción de etiquetas de figuras al inglés

Todos los módulos que generan figuras fueron actualizados para reemplazar
cualquier texto visible en español (títulos, etiquetas de ejes, leyendas,
anotaciones de colorbar) por su equivalente en inglés, en cumplimiento
de los requisitos de la revista destino.

Módulos modificados:

| Módulo | Figuras de salida |
|---|---|
| `src/loading.py` | `T1.1.3_station_coverage.png` |
| `src/missing.py` | `missing_matrix.png`, `missing_by_state_year.png`, `missing_biserial.png`, `missing_dropout.png`, `missing_dropout_temporal.png` |
| `src/distributions.py` | `dist_histogram.png`, `dist_seasonal_boxplot.png`, `dist_pci.png`, `dist_state_month_clustermap.png`, `dist_stl.png`, `dist_variogram.png` |
| `src/spatial.py` | `map_coverage.png`, `map_kriging.png`, `map_seasonal.png` |
| `src/anomalies.py` | `anomaly_capa1_summary.png`, `anomaly_capa2_summary.png`, `anomaly_capa3_summary.png`, `anomaly_capa4_summary.png` |
| `src/consolidation.py` | `anomaly_consolidation_summary.png` |
| `src/coda_prep.py` | `coda_subset_selection.png`, `coda_zero_treatment.png` |
| `src/compositional.py` | `coda_logratio_transforms.png` |
| `src/clustering.py` | `clustering_diagnostics.png` |
| `src/validation.py` | `method_concordance.png`, `regime_maps.png`, `cluster_profiles_kmeans.png` |
| `src/voronoi_map.py` | `mapa_voronoi_k28.png` |

#### B. Actualización de resolución de figuras a 900 DPI

Todas las llamadas a `fig.savefig()` fueron actualizadas de los valores
originales (130, 140 o 150 DPI) a **900 DPI**, cumpliendo el estándar
mínimo requerido para figuras de revista en formato impreso.

#### C. Script de ejecución del pipeline

Se añadió `run_all.sh` en la raíz del proyecto. Ejecuta el pipeline
completo T1–T3.5 en orden de dependencias mediante `conda run -n lluvia`,
sin requerir activación previa del entorno:

```bash
./run_all.sh
```

---

## Referencias Clave

| ID | Referencia | Relevancia |
|---|---|---|
| [1] | Aitchison (1986). *The Statistical Analysis of Compositional Data*. Chapman & Hall. | Fundamento teórico CoDA |
| [2] | Filzmoser, Hron & Templ (2018). *Applied Compositional Data Analysis*. Springer. | Implementación moderna CoDA |
| [3] | Palarea-Albaladejo & Martín-Fernández (2008). Computers & Geosciences, 34(8). | lrEM para ceros |
| [4] | Martín-Fernández et al. (2003). Mathematical Geology, 35(3). | Reemplazo multiplicativo |
| [5] | Hubert & Vandervieren (2008). CSDA, 52(12). | Adjusted boxplot |
| [6] | Anselin (1995). Geographical Analysis, 27(2). | LISA / Moran local |
| [7] | Tibshirani et al. (2001). JRSS-B, 63(2). | Gap statistic |
| [8] | Hennig (2007). CSDA, 52(1). | Bootstrap Jaccard |
| [9] | Liu, Ting & Zhou (2008). ICDM. | Isolation Forest |
| [10] | García (2004). *Modificaciones al sistema de clasificación climática de Köppen*. UNAM. | Referencia climatológica México |
