#!/usr/bin/env bash
# Publish a verified conversion and print the immutable model.yaml source block.
# Usage: publish_model_conversion.sh <model_dir> <conversion> <export_dir>
# Requires the staging publisher AWS role and GITHUB_RUN_ID, GITHUB_RUN_ATTEMPT, GITHUB_SHA.
set -euo pipefail
[[ $# -eq 3 ]] || { echo "usage: $0 <model_dir> <conversion> <export_dir>" >&2; exit 2; }
model_dir="$1"
conversion="$2"
export_dir="$3"
model_id="$(basename "$model_dir")"
model="$export_dir/model.onnx"
sha="$(sha256sum "$model" | cut -d ' ' -f 1)"
key="models-onnx/$model_id/$sha/${GITHUB_RUN_ID:?}-${GITHUB_RUN_ATTEMPT:?}/model.onnx"
aws s3api put-object --bucket sima-neat-artifacts-staging --key "$key" --body "$model" \
  --if-none-match '*' --server-side-encryption aws:kms --ssekms-key-id alias/sima-neat-artifacts-staging \
  --checksum-sha256 "$(openssl dgst -sha256 -binary "$model" | base64 -w0)" \
  --metadata "git-sha=${GITHUB_SHA:?},config=$conversion"
uri="https://artifacts.stg.neat.sima.ai/$key"
downloaded="$(mktemp)"
trap 'rm -f "$downloaded"' EXIT
curl --fail --silent --show-error "$uri" -o "$downloaded"
echo "$sha  $downloaded" | sha256sum --check
"${REGISTRY_PYTHON:-python3}" -m model_registry conversion-source "$model_dir" \
  --model "$model" --uri "$uri" --conversion "$conversion" > "$export_dir/source.yaml"
cat "$export_dir/source.yaml"
if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
  printf '```yaml\n' >> "$GITHUB_STEP_SUMMARY"
  cat "$export_dir/source.yaml" >> "$GITHUB_STEP_SUMMARY"
  printf '```\n' >> "$GITHUB_STEP_SUMMARY"
fi
