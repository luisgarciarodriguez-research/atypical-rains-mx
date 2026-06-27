"""
Constantes globales del proyecto de precipitación atípica en México.

Define rutas canónicas del árbol de directorios, parámetros del dataset
(código de faltante, separador, columnas de metadatos), constantes temporales
(rango 2013–2026), umbrales de detección de anomalías (z-score, percentiles,
LOF, Isolation Forest, Mahalanobis) y configuración del clustering composicional.
Todas las rutas se resuelven de forma relativa a la raíz del repositorio.

Proyecto : Detección de Precipitación Atípica en México (2013–2026)
Unidad   : Universidad Nacional Autónoma de México (UNAM)
           Instituto de Investigaciones en Matemáticas Aplicadas y en Sistemas (IIMAS)
           Posgrado en Ciencia e Ingeniería de la Computación (PCIC)
Investigador líder : Dr. José Antonio Neme Castillo
Contribuidor       : Luis García Rodríguez
"""
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
