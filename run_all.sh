#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

conda run -n lluvia --no-capture-output python -m src.loading      && \
conda run -n lluvia --no-capture-output python -m src.missing      && \
conda run -n lluvia --no-capture-output python -m src.distributions && \
conda run -n lluvia --no-capture-output python -m src.spatial      && \
conda run -n lluvia --no-capture-output python -m src.anomalies    && \
conda run -n lluvia --no-capture-output python -m src.consolidation && \
conda run -n lluvia --no-capture-output python -m src.coda_prep    && \
conda run -n lluvia --no-capture-output python -m src.compositional && \
conda run -n lluvia --no-capture-output python -m src.clustering   && \
conda run -n lluvia --no-capture-output python -m src.validation   && \
conda run -n lluvia --no-capture-output python -m src.voronoi_map  && \
conda run -n lluvia --no-capture-output python -m src.report       && \
conda run -n lluvia --no-capture-output python -m src.slides

echo ""
echo "Pipeline completado. Figuras en outputs/figures/"
