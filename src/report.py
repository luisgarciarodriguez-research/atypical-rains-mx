"""
Generador del reporte técnico final en PDF — Análisis de Precipitación Atípica en México.

Ensambla con ReportLab un documento académico de múltiples secciones que integra
todos los resultados del pipeline (T1–T3.5):

  Sección 1  Resumen ejecutivo: alcance, fuente de datos (SMN), período 2013–2026
             y hallazgos principales.
  Sección 2  Calidad de datos: estadísticas de completitud (46.2 % faltante),
             diagnóstico MCAR y mapa de cobertura de estaciones.
  Sección 3  Distribuciones y tendencias: histogramas, STL, Mann-Kendall y
             variogramas empíricos.
  Sección 4  Catálogo de anomalías: tabla de consenso multi-capa (T2.1–T2.5),
             kappa de Fleiss y mapas de densidad.
  Sección 5  Regímenes pluviométricos: perfiles composicionales CLR/ILR,
             mapas de clusters y etiquetas climatológicas (T3.3–T3.5).
  Sección 6  Metodología: descripción del análisis composicional (CoDA),
             clustering en espacio Aitchison y validación cruzada.

La salida se escribe en ``outputs/reports/reporte_final_lluvias_atipicas_mx.pdf``.
Punto de entrada: ``build_report(verbose=True)``.

Proyecto : Detección de Precipitación Atípica en México (2013–2026)
Unidad   : Universidad Nacional Autónoma de México (UNAM)
           Instituto de Investigaciones en Matemáticas Aplicadas y en Sistemas (IIMAS)
           Posgrado en Ciencia e Ingeniería de la Computación (PCIC)
Investigador líder : Dr. José Antonio Neme Castillo
Contribuidor       : Luis García Rodríguez
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image as PILImage

# reportlab
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    BaseDocTemplate, Frame, Image, NextPageTemplate, PageBreak,
    PageTemplate, Paragraph, SimpleDocTemplate, Spacer, Table,
    TableStyle, HRFlowable, KeepTogether,
)
from reportlab.platypus.flowables import HRFlowable

from src.config import DATA_PROCESSED, DATA_CATALOGS, FIGURES, ROOT

OUTPUT_PDF = ROOT / "outputs" / "reports" / "reporte_final_lluvias_atipicas_mx.pdf"

# ── Paleta ────────────────────────────────────────────────────────────────────
C_AZUL      = colors.HexColor("#1565C0")
C_AZUL_CLARO= colors.HexColor("#BBDEFB")
C_GRIS      = colors.HexColor("#616161")
C_GRIS_CLARO= colors.HexColor("#F5F5F5")
C_ROJO      = colors.HexColor("#C62828")
C_VERDE     = colors.HexColor("#2E7D32")
C_NARANJA   = colors.HexColor("#E65100")
C_BLANCO    = colors.white
C_NEGRO     = colors.black

PAGE_W, PAGE_H = A4
MARGIN = 2.2 * cm
USABLE_W = PAGE_W - 2 * MARGIN

# ── Estilos ───────────────────────────────────────────────────────────────────

def _make_styles() -> dict:
    base = getSampleStyleSheet()
    def s(name, **kw):
        return ParagraphStyle(name, **kw)

    return {
        "portada_titulo": s("portada_titulo",
            fontSize=22, leading=28, alignment=TA_CENTER,
            textColor=C_AZUL, fontName="Helvetica-Bold", spaceAfter=12),
        "portada_subtitulo": s("portada_subtitulo",
            fontSize=13, leading=18, alignment=TA_CENTER,
            textColor=C_GRIS, fontName="Helvetica", spaceAfter=8),
        "portada_meta": s("portada_meta",
            fontSize=10, leading=14, alignment=TA_CENTER,
            textColor=C_GRIS, fontName="Helvetica"),
        "h1": s("h1",
            fontSize=14, leading=18, fontName="Helvetica-Bold",
            textColor=C_AZUL, spaceBefore=16, spaceAfter=6),
        "h2": s("h2",
            fontSize=11, leading=15, fontName="Helvetica-Bold",
            textColor=C_GRIS, spaceBefore=10, spaceAfter=4),
        "h3": s("h3",
            fontSize=10, leading=13, fontName="Helvetica-BoldOblique",
            textColor=C_AZUL, spaceBefore=8, spaceAfter=3),
        "body": s("body",
            fontSize=9.5, leading=14, fontName="Helvetica",
            textColor=C_NEGRO, alignment=TA_JUSTIFY, spaceAfter=6),
        "bullet": s("bullet",
            fontSize=9, leading=13, fontName="Helvetica",
            textColor=C_NEGRO, leftIndent=14, spaceAfter=2,
            bulletIndent=4),
        "caption": s("caption",
            fontSize=8, leading=11, fontName="Helvetica-Oblique",
            textColor=C_GRIS, alignment=TA_CENTER, spaceBefore=3, spaceAfter=8),
        "table_header": s("table_header",
            fontSize=8.5, leading=11, fontName="Helvetica-Bold",
            textColor=C_BLANCO, alignment=TA_CENTER),
        "table_cell": s("table_cell",
            fontSize=8, leading=11, fontName="Helvetica",
            textColor=C_NEGRO, alignment=TA_LEFT),
        "table_cell_c": s("table_cell_c",
            fontSize=8, leading=11, fontName="Helvetica",
            textColor=C_NEGRO, alignment=TA_CENTER),
        "note": s("note",
            fontSize=8, leading=11, fontName="Helvetica-Oblique",
            textColor=C_GRIS, leftIndent=8, spaceAfter=4),
        "ref": s("ref",
            fontSize=8.5, leading=12, fontName="Helvetica",
            textColor=C_NEGRO, leftIndent=16, firstLineIndent=-16, spaceAfter=4),
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fig(path: Path, width: float = USABLE_W, caption: str = "") -> list:
    """Devuelve [Image, Paragraph(caption)] si el archivo existe."""
    if not path.exists():
        return [Paragraph(f"[Figura no encontrada: {path.name}]",
                          _make_styles()["note"])]
    S = _make_styles()
    with PILImage.open(path) as pil_img:
        orig_w, orig_h = pil_img.size
    height = width * (orig_h / orig_w)
    img = Image(str(path), width=width, height=height)
    items = [img]
    if caption:
        items.append(Paragraph(caption, S["caption"]))
    return items


def _hr(color=C_AZUL_CLARO, thickness=1) -> HRFlowable:
    return HRFlowable(width="100%", thickness=thickness, color=color,
                      spaceAfter=4, spaceBefore=4)


def _df_to_table(
    df: pd.DataFrame,
    col_widths: list | None = None,
    zebra: bool = True,
) -> Table:
    S = _make_styles()
    header = [Paragraph(str(c), S["table_header"]) for c in df.columns]
    rows   = [header]
    for _, row in df.iterrows():
        rows.append([Paragraph(str(v), S["table_cell_c"]) for v in row.values])

    tbl = Table(rows, colWidths=col_widths, repeatRows=1)
    style_cmds = [
        ("BACKGROUND",  (0, 0), (-1, 0), C_AZUL),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [C_GRIS_CLARO, C_BLANCO] if zebra else [C_BLANCO]),
        ("GRID",        (0, 0), (-1, -1), 0.4, colors.HexColor("#BDBDBD")),
        ("TOPPADDING",  (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING",(0, 0), (-1, -1), 4),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
    ]
    tbl.setStyle(TableStyle(style_cmds))
    return tbl


# ── Secciones del reporte ─────────────────────────────────────────────────────

def _portada(S: dict, today: str) -> list:
    items: list = []
    items.append(Spacer(1, 3.5 * cm))
    items.append(Paragraph(
        "Análisis de Precipitación Atípica en México", S["portada_titulo"]))
    items.append(Paragraph(
        "Detección de Anomalías y Clasificación de Regímenes Pluviométricos<br/>"
        "(2013–2026)", S["portada_subtitulo"]))
    items.append(Spacer(1, 0.5 * cm))
    items.append(_hr(C_AZUL, 2))
    items.append(Spacer(1, 0.5 * cm))
    items.append(Paragraph(
        "Departamento de Ciencias de la Información y la Computación<br/>"
        "Universidad Nacional Autónoma de México", S["portada_meta"]))
    items.append(Spacer(1, 0.3 * cm))
    items.append(Paragraph(f"Fecha de generación: {today}", S["portada_meta"]))
    items.append(Spacer(1, 1.5 * cm))

    # Cuadro de resumen rápido
    resumen_data = [
        ["Métrica", "Valor"],
        ["Estaciones pluviométricas", "1,959"],
        ["Período analizado", "Ene 2013 – May 2026 (161 meses)"],
        ["Código de dato faltante", "−99.0  (46.2% del total)"],
        ["Anomalías confirmadas (≥2 capas)", "14,290 celdas"],
        ["Subconjunto CoDA", "1,302 estaciones"],
        ["Regímenes identificados (K*)", "14 (Gap 1SE)"],
    ]
    col_w = [USABLE_W * 0.45, USABLE_W * 0.55]
    hdr   = [Paragraph(c, S["table_header"]) for c in resumen_data[0]]
    body  = [[Paragraph(str(v), S["table_cell_c"]) for v in r]
             for r in resumen_data[1:]]
    tbl   = Table([hdr] + body, colWidths=col_w, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0), C_AZUL),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [C_GRIS_CLARO, C_BLANCO]),
        ("GRID",         (0, 0), (-1, -1), 0.4, colors.HexColor("#BDBDBD")),
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
        ("LEFTPADDING",  (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("FONTSIZE",     (0, 1), (-1, -1), 9),
    ]))
    items.append(KeepTogether([tbl]))
    items.append(PageBreak())
    return items


def _seccion_1(S: dict) -> list:
    items: list = []
    items.append(Paragraph("1. Descripción del Dataset y Exploración", S["h1"]))
    items.append(_hr())
    items.append(Paragraph(
        "La base de datos consiste en registros mensuales de 1,959 estaciones "
        "pluviométricas del Sistema Meteorológico Nacional (SMN), distribuidas en "
        "los 31 estados de México y la CDMX. El período analizado abarca de enero "
        "de 2013 a mayo de 2026, generando una matriz de 1,959 × 161 celdas "
        "(315,399 observaciones potenciales).", S["body"]))
    items.append(Paragraph(
        "El 46.2% de las celdas presentan el código de faltante (−99.0), con "
        "patrones de omisión no aleatorios: estaciones del norte presentan mayor "
        "frecuencia de faltante en meses invernales, mientras que cierres operativos "
        "generan caídas abruptas (dropout) detectables temporalmente.", S["body"]))

    items.append(Paragraph("1.1  Cobertura y datos faltantes", S["h2"]))
    items += _fig(FIGURES / "T1.1.3_station_coverage.png", USABLE_W * 0.75,
                  "Fig. 1.1 — Distribución geográfica de estaciones (n=1,959). "
                  "Color y tamaño proporcionales al porcentaje de datos presentes.")
    items.append(Spacer(1, 0.3 * cm))
    items += _fig(FIGURES / "missing_matrix.png", USABLE_W,
                  "Fig. 1.2 — Matriz de faltantes por estación × mes (muestra aleatoria). "
                  "Franjas verticales indican periodos de dropout operativo.")

    items.append(Paragraph("1.2  Distribución estadística de la precipitación", S["h2"]))
    items += _fig(FIGURES / "dist_histogram.png", USABLE_W * 0.85,
                  "Fig. 1.3 — Distribución general de precipitación mensual (mm). "
                  "Marcada asimetría positiva (sesgada a la derecha); eje log para visualizar "
                  "la cola de eventos extremos.")
    items.append(Spacer(1, 0.3 * cm))
    items += _fig(FIGURES / "dist_seasonal_boxplot.png", USABLE_W * 0.85,
                  "Fig. 1.4 — Boxplots mensuales (todos los años, todas las estaciones). "
                  "El patrón monzónico Jun–Sep domina a nivel nacional.")

    items.append(Paragraph("1.3  Análisis espacial y semivariograma", S["h2"]))
    items += _fig(FIGURES / "dist_variogram.png", USABLE_W * 0.7,
                  "Fig. 1.5 — Semivariograma empírico y modelo esférico ajustado "
                  "(psill=6,719.85 mm², range=31.61°, nugget=1,440.91 mm²). "
                  "Los parámetros se usaron para kriging residual en Capa 3.")

    items.append(PageBreak())
    return items


def _seccion_2(S: dict) -> list:
    items: list = []
    items.append(Paragraph("2. Detección de Anomalías (T2)", S["h1"]))
    items.append(_hr())
    items.append(Paragraph(
        "Se implementó un sistema de detección en cuatro capas independientes, "
        "cada una enfocada en un aspecto distinto de la anomalía pluviométrica. "
        "La confirmación de una anomalía requiere al menos 2 capas activas.", S["body"]))

    # Tabla resumen de capas
    capas_data = pd.DataFrame([
        ["Capa 1", "Valores extremos\n(z-score + ADJ)", "1,634",   "0.52%",  "Artefactos de medición"],
        ["Capa 2", "Outliers de Hampel\n(MAD robusta)",  "37,026",  "11.74%", "Outliers locales en el tiempo"],
        ["Capa 3", "Anomalías espaciales\n(Kriging CV + LOF + LISA)", "14,906", "4.73%", "Inconsistencias espaciales"],
        ["Capa 4", "Anomalías multivariadas\n(IsoForest + AE + MCD)",  "33,780", "10.71%","Perfiles anuales anómalos"],
        ["Consolidado\n(≥2 capas)", "—", "14,290", "4.53%", "Catálogo final de anomalías"],
    ], columns=["Capa", "Método", "Flags", "% del dataset", "Interpretación"])

    items.append(Spacer(1, 0.2 * cm))
    col_w = [USABLE_W*0.1, USABLE_W*0.28, USABLE_W*0.1, USABLE_W*0.13, USABLE_W*0.39]
    items.append(KeepTogether([
        _df_to_table(capas_data, col_widths=col_w),
        Paragraph("Tabla 2.1 — Resumen de flags por capa de detección.", S["caption"]),
    ]))
    items.append(Spacer(1, 0.4 * cm))

    items.append(Paragraph("2.1  Capa 1 — Valores Extremos Univariados", S["h2"]))
    items += _fig(FIGURES / "anomaly_capa1_summary.png", USABLE_W,
                  "Fig. 2.1 — Capa 1: distribución de z-scores y flags por mes/estado. "
                  "Umbral |z| > 3.5 con corrección de Hampel ajustada.")
    items.append(Spacer(1, 0.3*cm))

    items.append(Paragraph("2.2  Capa 2 — Outliers de Hampel Ajustado (MAD Robusta)", S["h2"]))
    items.append(Paragraph(
        "Método de Hampel modificado con corrección por medcouple (Brys et al., 2004) "
        "para distribuciones asimétricas, aplicado por estación a la serie temporal "
        "de 161 meses. Un bug inicial (importación incorrecta de <i>medcouple_1d</i> "
        "en statsmodels 0.14.6) fue identificado y corregido.", S["body"]))
    items += _fig(FIGURES / "anomaly_capa2_summary.png", USABLE_W,
                  "Fig. 2.2 — Capa 2: distribución temporal y espacial de flags Hampel.")
    items.append(Spacer(1, 0.3*cm))

    items.append(Paragraph("2.3  Capa 3 — Anomalías Espaciales", S["h2"]))
    items.append(Paragraph(
        "Tres sub-métodos complementarios: (a) Residuales de kriging ordinario con "
        "validación cruzada de 5 pliegues (umbral 2.5σ); (b) LOF en espacio "
        "(Lat, Lon, precipitación), k=20; (c) LISA (Moran Local), k=8 vecinos, "
        "permutaciones=199. Se flagearon clusters Bajo-Bajo y Alto-Alto "
        "estadísticamente significativos (p<0.05).", S["body"]))
    items += _fig(FIGURES / "anomaly_capa3_summary.png", USABLE_W,
                  "Fig. 2.3 — Capa 3: distribución de anomalías espaciales por método y mes.")
    items.append(Spacer(1, 0.3*cm))

    items.append(Paragraph("2.4  Capa 4 — Anomalías Multivariadas (Perfil Anual)", S["h2"]))
    items.append(Paragraph(
        "Análisis del perfil anual de 12 meses completos por estación-año "
        "(6,642 perfiles de 2013–2025, 32 estados). Tres sub-métodos: "
        "(a) Isolation Forest (n=300 árboles, log1p); "
        "(b) Autoencoder 12→6→3→6→12 (ReLU, umbral en P95 del MSE); "
        "(c) MCD robusta por estado (Mahalanobis, α=0.025).", S["body"]))
    items += _fig(FIGURES / "anomaly_capa4_summary.png", USABLE_W,
                  "Fig. 2.4 — Capa 4: anomalías multivariadas en perfiles anuales.")
    items.append(Spacer(1, 0.3*cm))

    items.append(Paragraph("2.5  Consolidación del Catálogo", S["h2"]))
    items.append(Paragraph(
        "Las anomalías confirmadas por ≥2 capas se clasificaron según prioridad: "
        "<b>artefacto</b> (Capa 1 + n≥2, acción: recodificar como NaN), "
        "<b>inconsistencia_espacial</b> (Capa 3), "
        "<b>evento_extremo</b> (Capa 2, no Capa 3) e "
        "<b>indeterminado</b>. "
        "El coeficiente κ de Fleiss entre capas fue 0.065 (leve), esperado "
        "para métodos complementarios.", S["body"]))
    items += _fig(FIGURES / "anomaly_consolidation_summary.png", USABLE_W,
                  "Fig. 2.5 — Consolidación: distribución por clasificación, solapamiento "
                  "entre capas (Upset plot) y catálogo final (14,290 anomalías).")

    items.append(PageBreak())
    return items


def _seccion_3(S: dict) -> list:
    items: list = []
    items.append(Paragraph("3. Análisis de Datos Composicionales (CoDA)", S["h1"]))
    items.append(_hr())
    items.append(Paragraph(
        "Los patrones estacionales de precipitación se modelaron como "
        "<i>composiciones</i> en el símplex S¹², utilizando el marco de Aitchison (1986). "
        "Cada estación queda representada por su distribución mensual relativa "
        "(proporción del total anual), eliminando el efecto confundidor de la magnitud "
        "absoluta y habilitando el análisis en el espacio euclídeo inducido por ILR.", S["body"]))

    items.append(Paragraph("3.1  Selección del Subconjunto Analítico", S["h2"]))
    items.append(Paragraph(
        "Criterios de inclusión: ≥10 meses con datos por año, ≥3 años válidos, "
        "coordenadas dentro del polígono de México, sin datos de precipitación "
        "totalmente ausentes. Los 634 artefactos confirmados fueron recodificados "
        "como NaN antes de calcular composiciones.", S["body"]))
    items += _fig(FIGURES / "coda_subset_selection.png", USABLE_W,
                  "Fig. 3.1 — Criterios de selección del subconjunto CoDA. "
                  "De 1,959 estaciones originales se retuvieron 1,302 (66.5%).")
    items.append(Spacer(1, 0.3*cm))

    items.append(Paragraph("3.2  Tratamiento de Ceros", S["h2"]))
    items.append(Paragraph(
        "El logaritmo requerido por CoDA es indefinido en cero. Se aplicó "
        "<b>reemplazo multiplicativo</b> (Martín-Fernández et al., 2003): "
        "δ = 0.65 × mín(precipitación > 0), ajustando los valores no-cero "
        "para mantener Σwⱼ = 1. Un análisis de sensibilidad comparó este "
        "resultado con el reemplazo Bayesiano-Laplace (δ = 0.65/12).", S["body"]))
    items += _fig(FIGURES / "coda_zero_treatment.png", USABLE_W,
                  "Fig. 3.2 — Tratamiento de ceros: distribución de ceros por mes, "
                  "efecto del reemplazo multiplicativo y comparación de métodos.")
    items.append(Spacer(1, 0.3*cm))

    items.append(Paragraph("3.3  Transformaciones Log-Ratio", S["h2"]))
    items.append(Paragraph(
        "Se implementaron dos transformaciones complementarias:", S["body"]))
    items.append(Paragraph(
        "• <b>CLR</b>: yⱼ = ln(wⱼ / g(w)), donde g(w) es la media geométrica. "
        "Rango CLR: [−9.73, +4.61]. Suma por fila: máx|Σ| = 2.3×10⁻¹⁴ ✓", S["bullet"]))
    items.append(Paragraph(
        "• <b>ILR-SBP</b>: base ortonormal construida a partir de la Partición Binaria "
        "Secuencial climatológica (11 particiones). ||Ψ Ψᵀ − I||_F = 5.2×10⁻¹⁶ ✓  "
        "Isometría verificada: error relativo máximo 1.8×10⁻¹⁵ ✓", S["bullet"]))
    items.append(Spacer(1, 0.2*cm))

    # Tabla de varianza ILR
    ilr_var_data = pd.DataFrame([
        ["ILR-1", "Secos vs Húmedos (Jun–Oct)", "28.1%", "28.1%"],
        ["ILR-2", "Invierno vs Prim. seca", "14.6%", "42.7%"],
        ["ILR-3", "Inicio vs Pleno húmedos", "15.8%", "58.6%"],
        ["ILR-4..7", "Contrastes intra-grupo", "17.7%", "76.3%"],
        ["ILR-8..11", "Refinamiento fino", "23.7%", "100.0%"],
    ], columns=["Coord.", "Interpretación SBP", "% Varianza", "% Acumulada"])
    items.append(KeepTogether([
        _df_to_table(ilr_var_data,
                     col_widths=[USABLE_W*0.1, USABLE_W*0.5, USABLE_W*0.2, USABLE_W*0.2]),
        Paragraph("Tabla 3.1 — Varianza por coordenada ILR-SBP.", S["caption"]),
    ]))
    items += _fig(FIGURES / "coda_logratio_transforms.png", USABLE_W,
                  "Fig. 3.3 — Transformaciones log-ratio: distribuciones CLR, "
                  "biplot ILR1 vs ILR2, varianza acumulada e isometría.")

    items.append(PageBreak())
    return items


def _seccion_4(S: dict) -> list:
    items: list = []
    items.append(Paragraph("4. Clustering en Espacio ILR", S["h1"]))
    items.append(_hr())
    items.append(Paragraph(
        "El clustering de los 1,302 perfiles composicionales se realizó en el "
        "espacio ILR (R¹¹), donde la distancia euclídea equivale exactamente a "
        "la distancia de Aitchison en el símplex. Se evaluaron K=2..15 con "
        "tres métodos.", S["body"]))

    items.append(Paragraph("4.1  Diagnóstico de K óptimo", S["h2"]))
    items.append(Paragraph(
        "Los criterios arrojaron señales distintas, consistentes con un "
        "gradiente climatológico continuo (no hay frontera abrupta):", S["body"]))

    k_diag = pd.DataFrame([
        ["Silhouette (K-Means)",    "K = 2",  "Estructura binaria dominante seco/húmedo"],
        ["Gap statistic (1SE)",     "K = 14", "Incremento monotónico; 1SE activa al final del rango"],
        ["GMM BIC",                 "K = 6",  "Mínimo claro; penaliza complejidad gaussiana"],
        ["Jaccard bootstrap K=14",  "0.632",  "Estabilidad moderada (0.60–0.75)"],
    ], columns=["Criterio", "K*", "Comentario"])
    items.append(KeepTogether([
        _df_to_table(k_diag,
                     col_widths=[USABLE_W*0.3, USABLE_W*0.12, USABLE_W*0.58]),
        Paragraph("Tabla 4.1 — Criterios de selección de K.", S["caption"]),
    ]))
    items.append(Spacer(1, 0.2*cm))
    items.append(Paragraph(
        "Se adoptó <b>K*=14</b> como criterio preestablecido en el protocolo "
        "(Gap 1SE), manteniendo comparabilidad entre los tres métodos. "
        "La nota metodológica recomienda explorar también K=6 (BIC) como "
        "solución parsimoniosa en análisis futuros.", S["body"]))
    items += _fig(FIGURES / "clustering_diagnostics.png", USABLE_W,
                  "Fig. 4.1 — Diagnóstico de clustering: codo + silhouette, "
                  "Calinski-Harabász, Gap statistic, BIC/AIC (GMM), "
                  "distribución Jaccard bootstrap, dendrograma truncado.")

    items.append(Paragraph("4.2  Concordancia inter-método", S["h2"]))
    ari_data = pd.DataFrame([
        ["K-Means",       "1.000", "0.659", "0.493"],
        ["Jerárquico (Ward)", "0.659", "1.000", "0.481"],
        ["GMM",           "0.493", "0.481", "1.000"],
    ], columns=["Método", "K-Means", "Jerárquico", "GMM"])
    nmi_data = pd.DataFrame([
        ["K-Means",           "1.000", "0.715", "0.640"],
        ["Jerárquico (Ward)", "0.715", "1.000", "0.635"],
        ["GMM",               "0.640", "0.635", "1.000"],
    ], columns=["Método", "K-Means", "Jerárquico", "GMM"])
    cw = [USABLE_W*0.36, USABLE_W*0.21, USABLE_W*0.21, USABLE_W*0.22]
    items.append(KeepTogether([
        Paragraph("ARI (Adjusted Rand Index):", S["h3"]),
        _df_to_table(ari_data, col_widths=cw),
        Spacer(1, 0.25*cm),
        Paragraph("NMI (Normalized Mutual Information):", S["h3"]),
        _df_to_table(nmi_data, col_widths=cw),
        Paragraph(
            "Tabla 4.2 — Matrices de concordancia. K-Means y Ward convergen "
            "sustancialmente (ARI 0.66, NMI 0.72). GMM captura estructura "
            "complementaria (modela densidades gaussianas en lugar de "
            "particiones voronoi).", S["caption"]),
    ]))
    items += _fig(FIGURES / "method_concordance.png", USABLE_W * 0.75,
                  "Fig. 4.2 — Heatmaps de ARI y NMI entre métodos.")

    items.append(PageBreak())
    return items


def _seccion_5(S: dict) -> list:
    items: list = []
    items.append(Paragraph("5. Regímenes Pluviométricos", S["h1"]))
    items.append(_hr())
    items.append(Paragraph(
        "Los 14 clusters identifican regímenes pluviométricos definidos por "
        "la forma del perfil composicional anual, independientemente de la "
        "magnitud absoluta de precipitación.", S["body"]))

    items.append(Paragraph("5.1  Distribución geográfica", S["h2"]))
    items += _fig(FIGURES / "regime_maps.png", USABLE_W,
                  "Fig. 5.1 — Distribución espacial de regímenes pluviométricos "
                  "para K-Means (izq.), Jerárquico Ward (centro) y GMM (der.). "
                  "Los tres métodos muestran coherencia geográfica: regímenes "
                  "de verano dominan el sur y Pacífico; el norte muestra regímenes "
                  "con mayor peso invernal.")

    items.append(Paragraph("5.2  Perfiles composicionales y etiquetas climatológicas", S["h2"]))
    items += _fig(FIGURES / "cluster_profiles_kmeans.png", USABLE_W,
                  "Fig. 5.2 — Perfiles composicionales de los 14 clusters (K-Means). "
                  "Barra = media mensual; banda sombreada = Q25–Q75. "
                  "Eje Y: proporción del total anual.")
    items.append(Spacer(1, 0.3*cm))

    # Tabla de etiquetas
    labels_data = pd.DataFrame([
        ["C1",  265, "Sep", 0.910, "70.8%",  "7.8%",  "bimodal (picos 5, 8)"],
        ["C2",   80, "Ago", 0.821, "80.0%", "11.4%",  "lluvias_verano (pico Ago)"],
        ["C3",  213, "Jul", 0.813, "85.6%",  "5.5%",  "lluvias_verano (pico Jul)"],
        ["C4",   43, "Ago", 0.712, "91.8%",  "4.1%",  "lluvias_verano (pico Ago)"],
        ["C5",  319, "Ago", 0.829, "81.5%",  "2.4%",  "lluvias_verano (pico Ago)"],
        ["C6",   18, "Mar", 0.876, "21.1%", "43.3%",  "lluvias_invierno (pico Mar)"],
        ["C7",   96, "Oct", 0.945, "60.2%", "17.4%",  "bimodal (picos 5, 9)"],
        ["C8",   18, "Ago", 0.762, "85.0%",  "8.1%",  "lluvias_verano (pico Ago)"],
        ["C9",  114, "Sep", 0.781, "86.6%",  "1.0%",  "bimodal (picos 5, 8)"],
        ["C10",  48, "Jul", 0.785, "88.6%",  "4.3%",  "lluvias_verano (pico Jul)"],
        ["C11",  11, "Sep", 0.722, "88.0%",  "6.0%",  "lluvias_verano (pico Sep)"],
        ["C12",  24, "Jun", 0.739, "89.1%",  "0.4%",  "bimodal (picos 5, 8)"],
        ["C13",  11, "Sep", 0.734, "89.2%",  "4.5%",  "lluvias_verano (pico Sep)"],
        ["C14",  42, "Sep", 0.751, "91.5%",  "1.7%",  "bimodal (picos 5, 8)"],
    ], columns=["Cluster", "n", "Mes pico", "H_rel", "% Verano", "% Invierno", "Etiqueta"])
    cw = [USABLE_W*0.09, USABLE_W*0.06, USABLE_W*0.1,
          USABLE_W*0.08, USABLE_W*0.1, USABLE_W*0.1, USABLE_W*0.47]
    items.append(KeepTogether([
        _df_to_table(labels_data, col_widths=cw),
        Paragraph(
            "Tabla 5.1 — Etiquetas climatológicas por cluster. "
            "H_rel: entropía de Shannon normalizada (0=concentrado, 1=uniforme). "
            "% Verano: proporción Jun–Oct; % Invierno: Dic–Feb.", S["caption"]),
    ]))
    items.append(Spacer(1, 0.3*cm))
    items.append(Paragraph(
        "<b>Patrones dominantes</b>: 9 de los 14 clusters presentan régimen de "
        "<i>lluvias de verano</i> (pico Jun–Sep, >80% en Jun–Oct), característico "
        "del monzón mexicano que afecta la vertiente Pacífico e interior. "
        "El cluster C6 (n=18) representa el régimen de <i>lluvias de invierno</i> "
        "(Dic–Feb), asociado con sistemas frontales en el noroeste. "
        "5 clusters exhiben patrón <i>bimodal</i> (mayo y septiembre), "
        "típico de zonas de transición o influencia del Golfo de México.", S["body"]))

    items.append(PageBreak())
    return items


def _seccion_6(S: dict) -> list:
    items: list = []
    items.append(Paragraph("6. Conclusiones", S["h1"]))
    items.append(_hr())
    items.append(Paragraph(
        "Este análisis integró técnicas de estadística robusta, geoestadística, "
        "análisis de datos composicionales y aprendizaje no supervisado para "
        "caracterizar la precipitación mensual en México (2013–2026).", S["body"]))

    concls = [
        ("<b>Anomalías detectadas</b>: el catálogo consolidado contiene 14,290 "
         "anomalías confirmadas (4.53% del dataset). La clasificación distingue "
         "634 artefactos de medición, 4,789 inconsistencias espaciales y "
         "eventos extremos climatológicamente plausibles."),
        ("<b>Calidad del dato</b>: el 46.2% de faltantes, combinado con 634 "
         "artefactos recodificados, subraya la necesidad de control de calidad "
         "explícito previo a cualquier modelado climático."),
        ("<b>Regímenes pluviométricos</b>: el análisis CoDA-ILR identificó "
         "K=14 regímenes pluviométricos con coherencia geográfica. El régimen "
         "dominante es el de lluvias de verano (monzón mexicano, Jun–Sep), con "
         "un cluster de lluvias invernales en el noroeste y patrones bimodales "
         "en la franja del Golfo."),
        ("<b>Metodología composicional</b>: el uso de ILR garantiza que los "
         "análisis de distancia y clustering operen en el espacio de Aitchison, "
         "evitando las inconsistencias de trabajar directamente con proporciones."),
        ("<b>Convergencia de métodos</b>: K-Means y jerárquico Ward presentan "
         "alta concordancia (ARI=0.66, NMI=0.72), validando la robustez de los "
         "regímenes identificados."),
    ]
    for c in concls:
        items.append(Paragraph(f"• {c}", S["bullet"]))
        items.append(Spacer(1, 0.15*cm))

    items.append(Spacer(1, 0.4*cm))
    items.append(Paragraph("Trabajo futuro", S["h2"]))
    futuro = [
        "Contraste de regímenes con clasificación Köppen-Geiger (Beck et al., 2023) "
        "y regiones hidrológicas CONAGUA mediante V de Cramér.",
        "Análisis de tendencias temporales (¿se desplazan los umbrales de inicio "
        "de estación lluviosa en el período 2013–2026?).",
        "Clustering año-por-año (Opción B) para evaluar estabilidad temporal de "
        "los regímenes identificados.",
        "Incorporación de variables atmosféricas (ENSO, PDO, AMO) como covariables "
        "explicativas de la asignación de cluster.",
    ]
    for f in futuro:
        items.append(Paragraph(f"• {f}", S["bullet"]))

    items.append(PageBreak())
    return items


def _referencias(S: dict) -> list:
    items: list = []
    items.append(Paragraph("Referencias", S["h1"]))
    items.append(_hr())
    refs = [
        "Aitchison, J. (1986). <i>The Statistical Analysis of Compositional Data</i>. "
        "Chapman & Hall. DOI: 10.1007/978-94-009-4109-0",
        "Beck, H.E. et al. (2023). High-resolution (1 km) Köppen-Geiger maps for "
        "1901–2099 based on constrained CMIP6 projections. <i>Scientific Data</i>, 10, 724.",
        "Brys, G., Hubert, M., & Struyf, A. (2004). A Robust Measure of Skewness. "
        "<i>Journal of Computational and Graphical Statistics</i>, 13(4), 996–1017.",
        "Hennig, C. (2007). Cluster-wise assessment of cluster stability. "
        "<i>Computational Statistics & Data Analysis</i>, 52(1), 258–271.",
        "Martín-Fernández, J.A., Barceló-Vidal, C., & Pawlowsky-Glahn, V. (2003). "
        "Dealing with zeros and missing values in compositional data sets. "
        "<i>Mathematical Geology</i>, 35(3), 253–278.",
        "Tibshirani, R., Walther, G., & Hastie, T. (2001). Estimating the number "
        "of clusters in a data set via the gap statistic. "
        "<i>Journal of the Royal Statistical Society B</i>, 63(2), 411–423.",
        "Van den Boogaart, K.G. & Tolosana-Delgado, R. (2013). "
        "<i>Analyzing Compositional Data with R</i>. Springer.",
    ]
    for i, r in enumerate(refs, 1):
        items.append(Paragraph(f"{i}. {r}", S["ref"]))
        items.append(Spacer(1, 0.1*cm))
    return items


# ── Header / Footer ───────────────────────────────────────────────────────────

def _make_header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(C_GRIS)
    # Header
    if doc.page > 1:
        canvas.drawString(MARGIN, PAGE_H - MARGIN + 0.4*cm,
                          "Análisis de Precipitación Atípica en México (2013–2026)")
        canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - MARGIN + 0.4*cm,
                               f"Pág. {doc.page}")
        canvas.setStrokeColor(C_AZUL_CLARO)
        canvas.setLineWidth(0.5)
        canvas.line(MARGIN, PAGE_H - MARGIN + 0.25*cm,
                    PAGE_W - MARGIN, PAGE_H - MARGIN + 0.25*cm)
    # Footer
    canvas.drawCentredString(PAGE_W / 2, MARGIN * 0.5,
                             f"UNAM — DCIC — {date.today().isoformat()}")
    canvas.restoreState()


# ── Punto de entrada ──────────────────────────────────────────────────────────

def run_report(verbose: bool = True) -> Path:
    """Genera el PDF final completo."""
    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    today = date.today().strftime("%d de %B de %Y")
    S = _make_styles()

    doc = SimpleDocTemplate(
        str(OUTPUT_PDF),
        pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN + 0.6*cm, bottomMargin=MARGIN,
        title="Análisis de Precipitación Atípica en México",
        author="UNAM DCIC",
    )

    story: list = []
    story += _portada(S, today)
    story += _seccion_1(S)
    story += _seccion_2(S)
    story += _seccion_3(S)
    story += _seccion_4(S)
    story += _seccion_5(S)
    story += _seccion_6(S)
    story += _referencias(S)

    doc.build(story, onFirstPage=_make_header_footer,
              onLaterPages=_make_header_footer)

    size_kb = OUTPUT_PDF.stat().st_size / 1024
    if verbose:
        print(f"[OK] PDF generado: {OUTPUT_PDF}")
        print(f"     Tamaño: {size_kb:.0f} KB")
    return OUTPUT_PDF


if __name__ == "__main__":
    run_report(verbose=True)
