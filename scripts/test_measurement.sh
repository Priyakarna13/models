#!/usr/bin/env bash
# Preserve a failed test's evidence for publication. Usage: test_measurement.sh <stage> <model_dir> <output>
set -euo pipefail
status="$3/test-status.txt"
scripts/run_model_tests.sh "$1" "$2" "$3/$1-result.json" > "$status" || [[ "$(<"$status")" == failed ]]
