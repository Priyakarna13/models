#!/usr/bin/env bash
# Publish a converted ONNX and its optional standard external weights.
# Usage: publish_model_conversion.sh <model_id> <conversion> <export_dir>
set -euo pipefail
[[ $# -eq 3 ]] || { echo "usage: $0 <model_id> <conversion> <export_dir>" >&2; exit 2; }
model_id="$1"
conversion="$2"
export_dir="$3"
files=(model.onnx)
[[ ! -f "$export_dir/model.onnx.ext" ]] || files+=(model.onnx.ext)
pin="$export_dir/source.yaml"
if (( ${#files[@]} > 1 )); then
  printf 'source:\n  main_file: model.onnx\n  files:\n' > "$pin"
fi
downloaded="$(mktemp "$export_dir/.published.XXXXXX")"
trap 'rm -f "$downloaded"' EXIT
for name in "${files[@]}"; do
  file="$export_dir/$name"
  sha="$(sha256sum "$file" | cut -d ' ' -f 1)"
  size="$(stat -c %s "$file")"
  key="models-onnx/$model_id/$sha/${GITHUB_RUN_ID:?}-${GITHUB_RUN_ATTEMPT:?}/$name"
  aws s3api put-object --bucket sima-neat-artifacts-staging --key "$key" --body "$file" \
    --if-none-match '*' --server-side-encryption aws:kms --ssekms-key-id alias/sima-neat-artifacts-staging \
    --checksum-sha256 "$(openssl dgst -sha256 -binary "$file" | base64 -w0)" \
    --metadata "git-sha=${GITHUB_SHA:?},config=$conversion"
  uri="https://artifacts.stg.neat.sima.ai/$key"
  curl --fail --silent --show-error "$uri" -o "$downloaded"
  echo "$sha  $downloaded" | sha256sum --check
  if (( ${#files[@]} == 1 )); then
    printf 'source:\n  type: file\n  uri: %s\n  sha256: %s\n  size_bytes: %s\n' "$uri" "$sha" "$size" > "$pin"
  else
    printf '    %s:\n      uri: %s\n      sha256: %s\n      size_bytes: %s\n' "$name" "$uri" "$sha" "$size" >> "$pin"
  fi
done
cat "$pin"
if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
  printf '```yaml\n%s\n```\n' "$(cat "$pin")" >> "$GITHUB_STEP_SUMMARY"
fi
