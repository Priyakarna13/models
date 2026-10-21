#!/usr/bin/env bash
# Publish one whole-model report. Usage: publish_measurement.sh <stage> <output>
set -euo pipefail
model-registry publish-report --manifest _measurement/manifest.json --result "$2/$1-result.json" \
  --api-url "$REGISTRY_API_URL" --aws-region "$AWS_REGION" --bucket "$ARTIFACT_BUCKET" \
  --kms-key-id "$KMS_KEY_ID" --environment staging \
  --github-run-url "$GITHUB_SERVER_URL/$GITHUB_REPOSITORY/actions/runs/$GITHUB_RUN_ID" \
  --test-status "$(<"$2/test-status.txt")"
