#!/usr/bin/env bash
# Bind hosted compile receipts and profile keys before the board runs. Usage: prepare_measurement.sh
set -euo pipefail
model-registry prepare-measurement --repository _source --compiled-dir _compiled \
  --model "$MODEL_ID" --build-profile "$BUILD_PROFILE_ID" \
  --branch "$SOURCE_BRANCH" --git-sha "$SOURCE_SHA" --runtime _measurement/runtime.json \
  --read-api-url "$REGISTRY_READ_API_URL" --api-url "$REGISTRY_API_URL" \
  --aws-region "$AWS_REGION" --output _measurement/manifest.json
