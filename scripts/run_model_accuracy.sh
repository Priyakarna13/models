#!/usr/bin/env bash
# Run one whole-model accuracy stage with the public local/CI command.
# Usage: run_model_accuracy.sh <model_dir> <build_profile> <compiled_dir> <output_dir>
set -euo pipefail
if [[ $# -ne 4 ]]; then
  echo "usage: $0 <model_dir> <build_profile> <compiled_dir> <output_dir>" >&2
  exit 2
fi
runtime_args=()
[[ -z "${MEASUREMENT_RUNTIME:-}" ]] || runtime_args=(--runtime "$MEASUREMENT_RUNTIME")
exec "${REGISTRY_PYTHON:-python3}" -m model_registry accuracy "$1" \
  --build-profile "$2" --compiled-dir "$3" --output "$4" "${runtime_args[@]}"
