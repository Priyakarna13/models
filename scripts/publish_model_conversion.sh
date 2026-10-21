#!/usr/bin/env bash
# Publish verified conversion output. Usage: publish_model_conversion.sh
set -euo pipefail
model="$EXPORT_DIR/model.onnx"
sha="$(sha256sum "$model" | cut -d ' ' -f 1)"
size="$(stat -c %s "$model")"
key="models-onnx/$MODEL_ID/$sha/$GITHUB_RUN_ID-$GITHUB_RUN_ATTEMPT/model.onnx"
aws s3api put-object --bucket sima-neat-artifacts-staging --key "$key" --body "$model" \
  --if-none-match '*' --server-side-encryption aws:kms --ssekms-key-id alias/sima-neat-artifacts-staging \
  --checksum-sha256 "$(openssl dgst -sha256 -binary "$model" | base64 -w0)" \
  --metadata "git-sha=$GITHUB_SHA,stage=conversion"
uri="https://artifacts.stg.neat.sima.ai/$key"
curl --fail --silent --show-error "$uri" -o "$RUNNER_TEMP/downloaded.onnx"
echo "$sha  $RUNNER_TEMP/downloaded.onnx" | sha256sum --check
printf 'source:\n  type: file\n  uri: %s\n  sha256: %s\n  size_bytes: %s\n' "$uri" "$sha" "$size" > "$EXPORT_DIR/source.yaml"
cat "$EXPORT_DIR/source.yaml"
# shellcheck disable=SC2016
printf '```yaml\n%s\n```\n' "$(cat "$EXPORT_DIR/source.yaml")" >> "$GITHUB_STEP_SUMMARY"
