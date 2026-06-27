"""
Paquete ``src`` — pipeline completo de detección de precipitación atípica.

Expone los módulos del proyecto de forma importable como paquete Python:

  config        Constantes globales, rutas y umbrales de detección.
  loading       T1.1 — Carga y limpieza del dataset SMN.
  missing       T1.2 — Diagnóstico de datos faltantes.
  distributions T1.3 — Distribuciones, tendencias y variograma.
  spatial       T1.4 — Análisis espacial y kriging exploratorio.
  anomalies     T2.1–T2.4 — Detección de anomalías por capas.
  consolidation T2.5 — Catálogo consolidado de anomalías.
  coda_prep     T3.1–T3.2 — Filtrado y reemplazo multiplicativo de ceros.
  compositional T3.3 — Transformaciones CLR e ILR (Aitchison).
  clustering    T3.4 — Clustering en espacio ILR.
  validation    T3.5 — Validación e interpretación climatológica.
  report        Generador del reporte técnico final (PDF).
  slides        Generador de la presentación de avances (PPTX).

Proyecto : Detección de Precipitación Atípica en México (2013–2026)
Unidad   : Universidad Nacional Autónoma de México (UNAM)
           Instituto de Investigaciones en Matemáticas Aplicadas y en Sistemas (IIMAS)
           Posgrado en Ciencia e Ingeniería de la Computación (PCIC)
Investigador líder : Dr. José Antonio Neme Castillo
Contribuidor       : Luis García Rodríguez
"""
