#!/usr/bin/env bash
# Retain workflow input JSON without evaluating it. Usage: write_measurement_selection.sh
set -euo pipefail
mkdir -p _measurement
printf '%s\n' "$PLANNED_RUNTIME" | python -m json.tool > _measurement/runtime.json
printf '%s\n' "$PLANNED_BUILDS" | python -m json.tool > _measurement/builds.json
