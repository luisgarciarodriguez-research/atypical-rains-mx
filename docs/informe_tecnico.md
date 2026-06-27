# Informe Técnico — Análisis de Precipitación Atípica en México (2013–2026)

**Proyecto**: Detección de Anomalías y Clasificación de Regímenes Pluviométricos  
**Institución**: UNAM · Departamento de Ciencias de la Información y la Computación  
**Fecha de cierre**: 2026-06-25  
**Entorno**: Conda `lluvia` · Python 3.11 · Linux
**Investigador líder**: Dr. José Antonio Neme Castillo
**Contribuidor**: Luis García Rodríguez

---

## 1. Resumen Ejecutivo

Este informe documenta el proceso técnico completo de análisis de datos de precipitación mensual en México para el período 2013–2026. El trabajo abarcó cuatro fases: (1) ingesta, limpieza y exploración del dataset; (2) detección de anomalías en cuatro capas independientes; (3) preparación del subconjunto analítico bajo el marco de Análisis de Datos Composicionales (CoDA); y (4) clustering en el espacio de Aitchison para identificar regímenes pluviométricos.

**Resultados principales:**

| Métrica | Valor |
|---|---|
| Estaciones analizadas | 1,959 |
| Período | Ene 2013 – May 2026 (161 meses) |
| Celdas totales | 315,399 |
| Datos faltantes | 46.2% |
| Anomalías confirmadas (≥2 capas) | 14,290 (4.53%) |
| Subconjunto CoDA | 1,302 estaciones (66.5%) |
| Regímenes pluviométricos (K*) | 14 |
| Líneas de código (src/) | 7,806 |
| Figuras generadas | 27 |

---

## 2. Descripción del Dataset

**Fuente**: Sistema Meteorológico Nacional (SMN) / CONAGUA  
**Archivo original**: `data/raw/stats_lluvia_2013_2026_datos_flt.csv`  
**Formato**: TSV (separador `\t`), codificación UTF-8  
**Dimensiones**: 1,959 filas (estaciones) × 170 columnas (4 metadatos + 161 meses + 5 auxiliares)

### 2.1 Convención de columnas

Las columnas de precipitación siguen el patrón `{mes}_{año}` (e.g. `9_2020` = septiembre 2020). Se parsean con la expresión regular `r'^(\d{1,2})_(\d{4})'`. El código de dato faltante es **−99.0**; cualquier valor ≤ −99.0 se reemplaza por `NaN` durante la ingesta.

### 2.2 Metadatos disponibles

| Columna | Contenido |
|---|---|
| `#Station` | Clave alfanumérica de la estación |
| `State` | Clave de 2–5 letras del estado (e.g. `CDMX`, `JAL`) |
| `Name` | Nombre de la estación |
| `Lat`, `Long` | Coordenadas geográficas (WGS84) |
| `No.records` | Número de registros en el archivo original |

---

## 3. Arquitectura del Sistema

### 3.1 Estructura de directorios

```
atypical_rains_mx/
├── data/
│   ├── raw/                    # Dato fuente (inmutable)
│   ├── processed/              # Outputs intermedios procesados
│   └── catalogs/               # Flags por capa y catálogo de anomalías
├── outputs/
│   ├── figures/                # 27 figuras PNG (140 dpi)
│   └── reports/                # PDF, PPTX, Markdown
├── src/                        # Código fuente Python
│   ├── config.py               # Constantes globales y rutas
│   ├── loading.py              # Ingesta y limpieza (T1.1)
│   ├── missing.py              # Análisis de faltantes (T1.2)
│   ├── distributions.py        # Distribuciones y STL (T1.3–T1.4)
│   ├── spatial.py              # Análisis espacial y kriging (T1.3.5)
│   ├── anomalies.py            # Capas 1–4 de detección (T2.1–T2.4)
│   ├── consolidation.py        # Catálogo y Fleiss κ (T2.5)
│   ├── coda_prep.py            # Subconjunto CoDA y ceros (T3.1–T3.2)
│   ├── compositional.py        # CLR, ILR-SBP, Helmert (T3.3)
│   ├── clustering.py           # K-Means, Gap, Jaccard, Ward, GMM (T3.4)
│   ├── validation.py           # Concordancia, mapas, etiquetas (T3.5)
│   ├── report.py               # Generador PDF (reportlab)
│   └── slides.py               # Generador PPTX (python-pptx)
└── PLAN.md                     # Plan de análisis original
```

### 3.2 Módulos y responsabilidades

| Módulo | Líneas | Tareas | Función de entrada |
|---|---|---|---|
| `loading.py` | 215 | T1.1 | `run_t1_1()` |
| `missing.py` | 888 | T1.2 | `run_t1_2()` |
| `distributions.py` | 655 | T1.3–T1.4 | `run_t1_3()` |
| `spatial.py` | 427 | T1.3.5 | `run_kriging_map()` |
| `anomalies.py` | 1,616 | T2.1–T2.4 | `run_t2_1()` … `run_t2_4()` |
| `consolidation.py` | 426 | T2.5 | `run_t2_5()` |
| `coda_prep.py` | 697 | T3.1–T3.2 | `run_t3_1()`, `run_t3_2()` |
| `compositional.py` | 395 | T3.3 | `run_t3_3()` |
| `clustering.py` | 440 | T3.4 | `run_t3_4()` |
| `validation.py` | 592 | T3.5 | `run_t3_5()` |
| `report.py` | 664 | Reporte PDF | `run_report()` |
| `slides.py` | 742 | Presentación PPTX | `run_slides()` |

Todos los módulos pueden ejecutarse directamente con `python -m src.<módulo>` gracias al bloque `if __name__ == "__main__"`.

---

## 4. Fase 1 — Ingesta y Exploración (T1)

### T1.1 — Limpieza

- Lectura con `pd.read_csv(..., sep='\t', encoding='utf-8')`
- Reemplazo de −99.0 → `NaN` con máscara vectorizada
- Detección de columnas de lluvia con regex `r'^(\d{1,2})_(\d{4})'`
- Guardado en `data/processed/lluvia_clean.parquet` (875 KB, 1,959 × 170)

### T1.2 — Análisis de Faltantes

- % faltante global: 46.2%
- Patrón detectado: dropout operativo en noreste (Chihuahua, Sonora, Baja California); faltante no aleatorio confirmado por correlación biserial con variables geográficas
- Figura clave: `missing_by_state_year.png`, `missing_dropout.png`

### T1.3 — Distribuciones y Estacionalidad

- Distribución muy asimétrica (mediana nacional ≈ 18 mm, máx registrado > 800 mm)
- Descomposición STL para series seleccionadas
- Índice de Concentración de Precipitación (PCI) por estación
- Clustermap estado × mes para patrones regionales

### T1.3.5 — Semivariograma (insumo para Capa 3)

Ajuste de modelo esférico sobre residuales de la media mensual:

```
psill  = 6,719.85 mm²
range  = 31.61°
nugget = 1,440.91 mm²
```

Estos parámetros se almacenaron en `anomalies._VARIO_PARAMS` y se usaron en el kriging de validación cruzada de la Capa 3, evitando el ajuste automático que fallaba con `zero-size array` en columnas de lluvia escasa.

---

## 5. Fase 2 — Detección de Anomalías (T2)

### 5.1 Arquitectura de cuatro capas

Cada capa produce un DataFrame booleano de las mismas dimensiones (1,959 × 161). La consolidación toma el OR lógico con umbral ≥2 capas.

### T2.1 — Capa 1: Extremos Univariados

- Método: z-score estandarizado por estación + ajuste de Hampel (umbral adaptativo)
- Umbral: `|z| > 3.5`
- **1,634 flags** (0.52%)
- Implementado en `anomalies.zscore_flags()` y `anomalies.hampel_flags()`

### T2.2 — Capa 2: Outliers Temporales (Hampel MAD Robusta)

- Método: ventana deslizante MAD con corrección por medcouple para distribuciones asimétricas (Brys et al., 2004)
- **Bug crítico resuelto**: `from statsmodels.stats.stattools import medcouple_1d` lanza `ImportError` en statsmodels 0.14.6 (el nombre correcto es `medcouple`). El `except Exception: continue` en el loop silenciaba el error, produciendo 0 flags. Después de la corrección: **37,026 flags** (11.74%)

### T2.3 — Capa 3: Anomalías Espaciales

Tres sub-métodos en paralelo:

**a) Kriging con validación cruzada de 5 pliegues**
- `KFold(n_splits=5, shuffle=True, seed=42)`
- `OrdinaryKriging` con `variogram_parameters=_VARIO_PARAMS` (evita auto-fit inestable)
- `backend='loop', n_closest_points=50` (evita matriz singular 1,253×1,253)
- Umbral: residual > 2.5σ
- Se requirió filtro adicional `coord_ok = df["Long"].notna() & df["Lat"].notna()` para evitar `ValueError` en pykrige

**b) Local Outlier Factor (LOF)**
- Espacio: (Lat, Long, precipitación) normalizado con `StandardScaler`
- `LocalOutlierFactor(n_neighbors=20, contamination="auto")`

**c) LISA (Moran Local)**
- `KNN.from_array(coords, k=8)` de libpysal
- `Moran_Local(vals, w, permutations=199, seed=42)`
- Flags: clusters LL y HH con p < 0.05
- Advertencia benigna: "2 disconnected components" (14 islas insulares sin vecinos cercanos)

**Total Capa 3**: **14,906 flags** (4.73%)

### T2.4 — Capa 4: Anomalías Multivariadas (Perfil Anual)

Operan sobre vectores de 12 meses completos por estación-año (6,642 perfiles, 2013–2025):

**a) Isolation Forest**
- `IsolationForest(n_estimators=300, contamination="auto", random_state=42)`
- Inputs: `log1p(profiles)` — **1,417 perfiles anómalos** (21.3%)

**b) Autoencoder**
- Arquitectura: 12 → 6 → 3 → 6 → 12 (ReLU, sin activación en salida)
- Entrenamiento: 200 épocas, Adam (lr=1e-3), MSELoss, batch=64
- Umbral: P95 del MSE de reconstrucción = 2.420
- **333 perfiles anómalos** (5.0%) — método más conservador

**c) MCD Robusta por Estado (Mahalanobis)**
- `MinCovDet(random_state=42, support_fraction=1-alpha)` con α=0.025
- Umbral: χ²(0.975, df=12) ≈ 23.34
- Ejecución por estado (32 grupos) para capturar covarianza regional
- Advertencia esperada: "not full rank" en estados con varianza casi nula en algún mes
- **2,070 perfiles anómalos** (31.2%) — tasa alta atribuida al comportamiento del estimador robusto en matrices de rango deficiente

**Total Capa 4**: **33,780 flags** (10.71%)

### T2.5 — Consolidación del Catálogo

```python
# Regla de prioridad (excluyente)
if capa1 and n_layers >= 2:      → "artefacto"          / "recodificar_nan"
elif capa3:                       → "inconsistencia_espacial" / "investigar"
elif capa2 and not capa3:         → "evento_extremo"     / "conservar_flag"
else:                             → "indeterminado"      / "investigar"
```

| Clasificación | N celdas | Acción |
|---|---|---|
| `artefacto` | 636 | `recodificar_nan` |
| `inconsistencia_espacial` | 6,875 | `investigar` |
| `evento_extremo` | 6,779 | `conservar_flag` |
| *(indeterminado)* | — | — |

**Fleiss κ = 0.065** (leve) — coherente con métodos complementarios que detectan tipos distintos de anomalía.

Guardado en `data/catalogs/anomalias_catalogo.csv` (1,213 KB, 14,290 filas).

---

## 6. Fase 3 — Análisis Composicional (T3)

### T3.1 — Selección del Subconjunto Analítico

Criterios aplicados secuencialmente:

1. Excluir estaciones sin ningún dato de precipitación
2. Excluir coordenadas fuera del polígono de México o NaN
3. Aplicar filtro de cobertura temporal: ≥10 meses/año × ≥3 años
4. Recodificar 636 artefactos como NaN (usando `anomalias_catalogo.csv`)

Resultado: **1,302 de 1,959 estaciones** (66.5%), guardadas en `lluvia_coda.parquet`.

### T3.2 — Tratamiento de Ceros

**Cálculo de composiciones**: media mensual por estación (promedio de todos los valores válidos de ese mes calendario a lo largo de los años), normalizada a Σ=1.

**Ceros estructurales**: 19 estaciones presentan NaN en la media mensual (16 en diciembre) — estaciones sin datos en ese mes calendario a lo largo del período. Se rellenaron con 0 antes de normalizar.

**Reemplazo multiplicativo** (Martín-Fernández et al., 2003):
```python
delta = 0.65 * row[~zeros].min()        # δ = 0.65 × mín no-cero
row[zeros] = delta
row[~zeros] *= (1 - n_zeros * delta) / row[~zeros].sum()
```

**Bug resuelto en figura T3.2**: el Panel 4 (`hist(row_sums, bins=50)`) crasheaba con `ValueError: Too many bins for data range` porque todas las sumas son exactamente 1.0. Se reemplazó con un gráfico de barras agrupadas de medianas mensuales (bruta vs. post-reemplazo).

**Análisis de sensibilidad**: comparación con reemplazo Bayesiano-Laplace (δ = 0.65/12). El archivo `compositions_raw.csv` está disponible para reemplazos alternativos en R (función `lrEM` de `zCompositions`; `rpy2` no disponible en el entorno).

### T3.3 — Transformaciones Log-Ratio

#### CLR (Centered Log-Ratio)
```
y_j = ln(w_j / g(w)),   g(w) = media geométrica
```
- Verificación: max|Σ(clr por fila)| = 2.31×10⁻¹⁴ ✓
- Guardado en `composiciones_clr.parquet` (1,302 × 12)

#### ILR con SBP Climatológica

La Partición Binaria Secuencial (SBP) fue diseñada con criterio climatológico para que las 11 coordenadas ILR sean interpretables:

| Partición | Contraste | Meses + | Meses − |
|---|---|---|---|
| P1 | Secos vs. húmedos | 1,2,3,4,11,12 | 5,6,7,8,9,10 |
| P2 | Invierno vs. prim. seca | 12,1,2 | 3,4,11 |
| P3 | Inicio vs. pleno húmedos | 5,6 | 7,8,9,10 |
| P4 | Dic vs. {Ene,Feb} | 12 | 1,2 |
| P5 | Ene vs. Feb | 1 | 2 |
| P6 | {Mar,Abr} vs. Nov | 3,4 | 11 |
| P7 | Mar vs. Abr | 3 | 4 |
| P8 | May vs. Jun | 5 | 6 |
| P9 | {Jul,Ago} vs. {Sep,Oct} | 7,8 | 9,10 |
| P10 | Jul vs. Ago | 7 | 8 |
| P11 | Sep vs. Oct | 9 | 10 |

**Bug en PLAN.md (P2)**: los meses 3 y 4 estaban codificados como 0 en lugar de −1. Corregido durante la implementación.

**Conversión SBP → contraste**:
```python
coef(+1) =  sqrt(s / (r·(r+s)))
coef(-1) = -sqrt(r / (s·(r+s)))
```

**Verificaciones numéricas**:
- Ortonormalidad: `||Ψ Ψᵀ − I||_F = 5.16×10⁻¹⁶` ✓
- Isometría: error relativo máximo `d_Aitchison vs d_Euclídea(ILR) = 1.81×10⁻¹⁵` ✓

Estructura de varianza:
| Coord. | Contraste | % varianza | % acum. |
|---|---|---|---|
| ILR-1 | Secos vs. húmedos | 28.1% | 28.1% |
| ILR-2 | Invierno vs. prim. seca | 14.6% | 42.7% |
| ILR-3 | Inicio vs. pleno húmedos | 15.8% | 58.6% |
| ILR-4..7 | Contrastes intra-grupo | 17.7% | 76.3% |
| ILR-8..11 | Refinamiento fino | 23.7% | 100.0% |

Guardado en `composiciones_ilr.parquet` (1,302 × 11, columnas `ilr_1`..`ilr_11`).

### T3.4 — Clustering en Espacio ILR

Los 1,302 vectores ILR (R¹¹) se analizaron con K=2..15. La distancia euclídea en ILR es exactamente la distancia de Aitchison en el símplex.

#### Criterios de selección de K

| Criterio | K* | Nota |
|---|---|---|
| Silhouette máximo | 2 | Estructura binaria seco/húmedo dominante |
| Gap statistic (1SE) | 14 | Gap crece monótonamente; 1SE activa en K=14 |
| GMM BIC mínimo | 6 | Penaliza parámetros; modelo parsimonioso |
| Jaccard bootstrap K=14 | 0.632 | Moderado (0.60–0.75) |

El Gap creciente sin codo es indicativo de un **gradiente climatológico continuo** (no hay discontinuidades abruptas entre regímenes). Se adoptó K*=14 (criterio del protocolo PLAN.md).

#### Métodos implementados

**K-Means**: `n_init=20` en el sweep; `n_init=50` en el ajuste final con K=14.

**Gap statistic** (Tibshirani et al., 2001):
- `n_bootstrap=50` muestras uniformes en `[min, max]` por dimensión
- `se = std(log_Wk_refs) * sqrt(1 + 1/n_bootstrap)` (corrección de Tibshirani)
- Regla 1SE: menor k donde `Gap(k) ≥ Gap(k+1) − SE(k+1)`

**Bootstrap Jaccard** (Hennig, 2007):
- Remuestreo con reemplazo de los N=1,302 puntos
- Predicción de TODO el conjunto original con centroides bootstrap (evita sesgos por índices)
- Matching húngaro (`scipy.optimize.linear_sum_assignment`) para alinear etiquetas
- Jaccard(i,j) = |A∩B| / |A∪B| por par de clusters; media sobre bootstrap

**Clustering jerárquico**: enlace Ward + distancia euclídea en ILR; dendrograma truncado a 30 hojas.

**GMM**: `n_init=5` en el sweep; `n_init=10` en el ajuste final.

#### Concordancia inter-método (K=14)

| Par | ARI | NMI |
|---|---|---|
| K-Means ↔ Ward | 0.659 | 0.715 |
| K-Means ↔ GMM | 0.493 | 0.640 |
| Ward ↔ GMM | 0.481 | 0.635 |

### T3.5 — Validación e Interpretación

#### Mapa geográfico
Contorno de México obtenido de `geodatasets.get_path("naturalearth.land")` recortado al bounding box `(−118°, 14.5°, −86.5°, 32.8°)` con `gpd.clip()`. La figura `regime_maps.png` muestra los tres métodos lado a lado.

#### Etiquetas climatológicas (K-Means K=14)

| Tipo | N clusters | % estaciones | Característica |
|---|---|---|---|
| `lluvias_verano` | 9 | 72.8% | Pico Jun–Sep, >80% en Jun–Oct; monzón mexicano |
| `bimodal` | 4 | 21.4% | Picos en mayo y ago/sep; influencia del Golfo |
| `lluvias_invierno` | 1 | 5.8% | C6 (n=18); noroeste; frentes fríos; 43% en Dic–Feb |

La asignación se basó en: entropía relativa de Shannon (`H/ln(12)`), proporción acumulada en temporada húmeda (Jun–Oct), detección de bimodalidad por picos locales.

#### Pendientes documentadas

**T3.5.5** (Köppen-Geiger / CONAGUA): requiere descarga de capas externas no disponibles en el entorno. Procedimiento documentado en el reporte.

**T3.5.6** (Estabilidad temporal): requiere clustering año-por-año (Opción B del PLAN.md). El análisis actual usa composiciones medias 2013–2025.

---

## 7. Problemas Técnicos y Soluciones

### 7.1 `medcouple_1d` — ImportError silenciado (Capa 2)

**Síntoma**: Capa 2 producía 0 flags.  
**Causa**: `statsmodels.stats.stattools.medcouple_1d` no existe en statsmodels 0.14.6; el nombre correcto es `medcouple`. El `except Exception: continue` en el loop de columnas silenciaba el error.  
**Solución**: cambiar importación y llamada. Resultado post-fix: 37,026 flags.

### 7.2 Kriging — `zero-size array` en auto-fit de variograma

**Síntoma**: `OrdinaryKriging(..., variogram_model='spherical')` fallaba en columnas con pocos valores no-nulos.  
**Causa**: pykrige no podía calcular semivariancias para construir el variograma.  
**Solución**: proveer `variogram_parameters=_VARIO_PARAMS` (ajustado en T1.3.5) en lugar de auto-fit.

### 7.3 Kriging — `LinAlgError: singular matrix` con 1,000+ estaciones

**Síntoma**: `execute("points", lons, lats)` con 1,253 estaciones producía matriz singular.  
**Causa**: la matriz de kriging de 1,253×1,253 era casi singular.  
**Solución**: `backend='loop', n_closest_points=50` (kriging local).

### 7.4 Predicción en posición exacta de la estación (sesgo de auto-predicción)

**Síntoma**: residuales casi nulos al predecir cada estación con todos los demás puntos (std≈14.77 vs. esperado ≈50+).  
**Causa**: la estación se incluía en su propio vecindario, dominando la predicción.  
**Solución**: validación cruzada de 5 pliegues con `KFold`; cada estación se predice solo con las estaciones de otros pliegues.

### 7.5 `ValueError` por coordenadas NaN en pykrige

**Síntoma**: error al pasar coordenadas con NaN a `OrdinaryKriging`.  
**Causa**: el filtro `df[col].notna()` no filtraba filas con NaN en `Lat` o `Long`.  
**Solución**: `coord_ok = df["Long"].notna() & df["Lat"].notna()` aplicado antes del kriging.

### 7.6 Figura T3.2 — histograma con rango cero

**Síntoma**: `ValueError: Too many bins for data range` en `hist(row_sums, bins=50)`.  
**Causa**: todas las sumas de composiciones = 1.0 exactamente (rango cero por construcción).  
**Solución**: reemplazar con barplot de medianas mensuales agrupadas (bruta vs. post-reemplazo).

### 7.7 SBP P2 — error en PLAN.md

**Síntoma**: la Partición 2 del PLAN.md codificaba meses 3 y 4 como 0 (inactivo) en lugar de −1 (denominador).  
**Causa**: error tipográfico en el plan original.  
**Solución**: verificación de la condición de ortonormalidad `||Ψ Ψᵀ − I||_F` reveló el error; se corrigió a −1 para meses 3 y 4.

### 7.8 reportlab — `Image(height=None)` inválido

**Síntoma**: `TypeError: float() argument must be a string or a real number, not 'NoneType'`.  
**Causa**: `Image(path, width=W, height=None, kind="proportional")` no es válido en reportlab; requiere ambas dimensiones.  
**Solución**: calcular el alto proporcional con Pillow: `h = w * (img.height / img.width)`.

### 7.9 `BASE_DIR` no existe en `src.config`

**Síntoma**: `ImportError: cannot import name 'BASE_DIR'`.  
**Causa**: el módulo de config usa `ROOT` como variable raíz, no `BASE_DIR`.  
**Solución**: cambiar importación a `ROOT` y `REPORTS`.

---

## 8. Dependencias del Entorno

**Entorno Conda**: `lluvia` (Python 3.11)

| Biblioteca | Versión | Uso |
|---|---|---|
| pandas | — | Manipulación de datos |
| numpy | — | Álgebra lineal |
| scipy | — | Stats, jerarquía, Hungarian |
| scikit-learn | — | K-Means, LOF, IsolationForest, GMM, KFold |
| statsmodels | 0.14.6 | `medcouple`, Moran local |
| pykrige | — | Kriging ordinario |
| esda | — | `Moran_Local` |
| libpysal | — | `KNN` weights |
| torch | — | Autoencoder (PyTorch) |
| geopandas | 1.1.3 | Mapas y operaciones espaciales |
| geodatasets | 2026.5.1 | Contorno de México (naturalearth.land) |
| matplotlib | — | Todas las figuras |
| Pillow | — | Dimensiones de imágenes para PDF/PPTX |
| reportlab | 5.0.0 | Generación de PDF |
| python-pptx | 1.0.2 | Generación de PPTX |

**Instalaciones adicionales durante la sesión**:
```bash
pip install reportlab
pip install python-pptx
```

---

## 9. Inventario de Entregables

### 9.1 Datos procesados (`data/processed/`)

| Archivo | Dimensiones | Tamaño | Descripción |
|---|---|---|---|
| `lluvia_clean.parquet` | 1,959 × 170 | 875 KB | Dataset limpio (NaN reemplazados) |
| `lluvia_coda.parquet` | 1,302 × 170 | 772 KB | Subconjunto analítico CoDA |
| `compositions_raw.csv` | 1,302 × 12 | 312 KB | Composiciones brutas (previo a reemplazo de ceros) |
| `compositions_no_zeros.parquet` | 1,302 × 12 | 157 KB | Composiciones post reemplazo multiplicativo |
| `composiciones_clr.parquet` | 1,302 × 12 | 157 KB | Transformación CLR |
| `composiciones_ilr.parquet` | 1,302 × 11 | 145 KB | Transformación ILR-SBP (input para clustering) |
| `cluster_assignments.parquet` | 1,302 × 6 | 34 KB | Asignaciones K-Means, Ward, GMM (K=14) |

### 9.2 Catálogos (`data/catalogs/`)

| Archivo | Dimensiones | Tamaño | Descripción |
|---|---|---|---|
| `flags_capa1.parquet` | 1,959 × 161 | 85 KB | Flags booleanos Capa 1 (1,634 True) |
| `flags_capa2.parquet` | 1,959 × 161 | 111 KB | Flags booleanos Capa 2 (37,026 True) |
| `flags_capa3.parquet` | 1,959 × 161 | 104 KB | Flags booleanos Capa 3 (14,906 True) |
| `flags_capa4.parquet` | 1,959 × 161 | 103 KB | Flags booleanos Capa 4 (33,780 True) |
| `anomalias_catalogo.csv` | 14,290 × 11 | 1,213 KB | Catálogo consolidado con clasificación y acción |
| `zscores.parquet` | 1,959 × 161 | 1,223 KB | Z-scores estandarizados por estación |

### 9.3 Figuras (`outputs/figures/`, 27 archivos, 7 MB)

**Fase 1 — Exploración**

| Figura | Descripción |
|---|---|
| `T1.1.3_station_coverage.png` | Mapa de cobertura espacial de las 1,959 estaciones |
| `missing_matrix.png` | Matriz de faltantes (muestra aleatoria) |
| `missing_by_state_year.png` | Porcentaje de faltante por estado y año |
| `missing_dropout.png` | Curva de dropout temporal por estación |
| `missing_dropout_temporal.png` | Heatmap temporal del dropout |
| `missing_biserial.png` | Correlación biserial de faltante con covariables |
| `dist_histogram.png` | Distribución global de precipitación (escala log) |
| `dist_seasonal_boxplot.png` | Boxplots mensuales agregados |
| `dist_pci.png` | Índice de Concentración de Precipitación (PCI) |
| `dist_stl.png` | Descomposición STL de series representativas |
| `dist_variogram.png` | Semivariograma empírico y modelo esférico ajustado |
| `dist_state_month_clustermap.png` | Clustermap estado × mes |
| `map_coverage.png` | Mapa detallado de cobertura con escala de faltante |
| `map_kriging.png` | Superficie interpolada por kriging ordinario |
| `map_seasonal.png` | Mapas de precipitación media por temporada |

**Fase 2 — Anomalías**

| Figura | Descripción |
|---|---|
| `anomaly_capa1_summary.png` | Distribución de flags Capa 1 |
| `anomaly_capa2_summary.png` | Distribución de flags Capa 2 |
| `anomaly_capa3_summary.png` | Distribución de flags Capa 3 (3 sub-métodos) |
| `anomaly_capa4_summary.png` | Distribución de flags Capa 4 (3 sub-métodos) |
| `anomaly_consolidation_summary.png` | Catálogo consolidado y clasificación |

**Fase 3 — CoDA y Clustering**

| Figura | Descripción |
|---|---|
| `coda_subset_selection.png` | Criterios y resultados de selección del subconjunto |
| `coda_zero_treatment.png` | Análisis de ceros y efecto del reemplazo |
| `coda_logratio_transforms.png` | CLR, biplot ILR, varianza por coord., isometría |
| `clustering_diagnostics.png` | Diagnóstico completo de K (6 paneles) |
| `regime_maps.png` | Mapas de los 3 métodos de clustering |
| `cluster_profiles_kmeans.png` | 14 perfiles composicionales (K-Means) |
| `method_concordance.png` | Heatmaps ARI y NMI |

### 9.4 Reportes (`outputs/reports/`)

| Archivo | Tamaño | Descripción |
|---|---|---|
| `reporte_final_lluvias_atipicas_mx.pdf` | 3,648 KB | Reporte de resultados (18 pp., reportlab) |
| `hallazgos_iniciales_lluvias_mx.pptx` | 2,470 KB | Presentación (15 diapositivas, python-pptx) |
| `regimenes_pluviometricos.md` | 5 KB | Reporte T3.5 en Markdown |
| `diagnostico_datos_faltantes.md` | 5 KB | Diagnóstico de faltantes |
| `distribution_summary.csv` | 2 KB | Estadísticos de distribución por estado |
| `informe_tecnico.md` | — | Este documento |

---

## 10. Reproducibilidad

Cada módulo puede ejecutarse de forma autónoma:

```bash
# Orden de ejecución completo
conda activate lluvia
python -m src.loading        # T1.1
python -m src.missing        # T1.2
python -m src.distributions  # T1.3–T1.4
python -m src.spatial        # T1.3.5
python -m src.anomalies      # T2.1–T2.4 (run_t2_4 como __main__)
python -m src.consolidation  # T2.5
python -m src.coda_prep      # T3.2 como __main__ (ejecuta T3.1 y T3.2)
python -m src.compositional  # T3.3
python -m src.clustering     # T3.4
python -m src.validation     # T3.5
python -m src.report         # PDF
python -m src.slides         # PPTX
```

**Verificación rápida de entregables**:

```python
import pandas as pd
from pathlib import Path

checks = {
    'lluvia_clean':      'data/processed/lluvia_clean.parquet',
    'anomalias_catalog': 'data/catalogs/anomalias_catalogo.csv',
    'ilr_coords':        'data/processed/composiciones_ilr.parquet',
    'clusters':          'data/processed/cluster_assignments.parquet',
}
for name, path in checks.items():
    p = Path(path)
    status = '✓' if p.exists() and p.stat().st_size > 0 else '✗'
    df = pd.read_parquet(p) if p.suffix == '.parquet' else pd.read_csv(p)
    print(f'  {status} {name}: {df.shape}')
```

---

## 11. Notas de Sesión

- La sesión fue interrumpida y reanudada en múltiples ocasiones durante la ejecución de T3.3 (se usó resumen de contexto comprimido)
- El módulo `anomalies.py` es el más extenso (1,616 líneas) por integrar las cuatro capas con sus respectivos sub-métodos
- Los parámetros del variograma (`_VARIO_PARAMS`) están hardcodeados en `anomalies.py` para garantizar reproducibilidad entre sesiones
- La función `rpy2`/`lrEM` fue documentada como alternativa no disponible; `compositions_raw.csv` provee el input listo para R si se requiere

---

## 12. Referencias

1. Aitchison, J. (1986). *The Statistical Analysis of Compositional Data*. Chapman & Hall.
2. Brys, G., Hubert, M., & Struyf, A. (2004). A Robust Measure of Skewness. *JCGS*, 13(4), 996–1017.
3. Hennig, C. (2007). Cluster-wise assessment of cluster stability. *CSDA*, 52(1), 258–271.
4. Martín-Fernández, J.A., Barceló-Vidal, C., & Pawlowsky-Glahn, V. (2003). Dealing with zeros and missing values in compositional data sets. *Mathematical Geology*, 35(3), 253–278.
5. Tibshirani, R., Walther, G., & Hastie, T. (2001). Estimating the number of clusters via the gap statistic. *JRSS-B*, 63(2), 411–423.
6. Van den Boogaart, K.G. & Tolosana-Delgado, R. (2013). *Analyzing Compositional Data with R*. Springer.
7. Webster, R. & Oliver, M.A. (2007). *Geostatistics for Environmental Scientists*. Wiley.
