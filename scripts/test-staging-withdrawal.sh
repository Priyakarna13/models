#!/usr/bin/env bash
# Usage: test-staging-withdrawal.sh
# Run in staging-promote-develop with a pending disposable promotion.
set -euo pipefail

promotion_id="${PROMOTION_ID:?Provide the reviewed pending promotion ID}"
model_id="${MODEL_ID:?Provide the reviewed model ID}"
profile_id="${PROFILE_ID:?Provide the reviewed profile ID}"
key="withdraw-test-${GITHUB_RUN_ID:?Run in the protected workflow}"
if [[ ! "$promotion_id" =~ ^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$ ]] ||
   [[ ! "$model_id" =~ ^[a-z0-9_]+$ || ! "$profile_id" =~ ^[a-z0-9_.-]+$ ]] ||
   [[ ! "$key" =~ ^[A-Za-z0-9_.:-]{1,160}$ ]]; then
  echo "Invalid promotion ID, model ID, profile ID, or idempotency key." >&2
  exit 2
fi

: "${AWS_ACCESS_KEY_ID:?Load the protected staging promotion role credentials}"
: "${AWS_SECRET_ACCESS_KEY:?Load the protected staging promotion role credentials}"
: "${AWS_SESSION_TOKEN:?Load the protected staging promotion role credentials}"
region="us-west-2"
actor="$(aws sts get-caller-identity --region "$region" --query Arn --output text)"
if [[ "$actor" != arn:aws:sts::987569578021:assumed-role/NeatStagingModelsRegistryPromotionRole/* ]]; then
  echo "Run with the staging Models promotion role from staging-promote-develop." >&2
  exit 1
fi

target="refs/heads/develop"
repository="/v1/repositories/sima-neat/models"
promotion_url="https://artifacts-registry.stg.neat.sima.ai${repository}/promotions/${promotion_id}"
catalog_base="https://models-registry.stg.neat.sima.ai${repository}/catalogs"
ref_query="git_ref=refs%2Fheads%2Fdevelop"
catalog_url="${catalog_base}/current?${ref_query}"

signed_request() {
  curl --silent --show-error --connect-timeout 10 --max-time 40 \
    --aws-sigv4 "aws:amz:${region}:execute-api" \
    --user "${AWS_ACCESS_KEY_ID}:${AWS_SECRET_ACCESS_KEY}" \
    --header "x-amz-security-token: ${AWS_SESSION_TOKEN}" "$@"
}

catalog_selection() {
  jq -er --arg model "$model_id" --arg profile "$profile_id" '
    [.catalog.document.models[] | select(.id == $model) | .approved_deployments[$profile].deployment_id]
    | map(select(. != null)) | first // ""
  '
}

promotion="$(signed_request --fail-with-body "$promotion_url")"
deployment_id="$(jq -er --arg target "$target" --arg model "$model_id" --arg profile "$profile_id" '
  select((.state == "pending" or .state == "promoted") and .target_git_ref == $target
    and .source.model_id == $model and .source.profile_id == $profile)
  | .deployment_id
' <<<"$promotion")"
before="$(curl --fail-with-body --silent --show-error --max-time 40 "$catalog_url")"
revision="$(jq -er '.catalog.catalog_revision | select(test("^[0-9a-f]{40}$"))' <<<"$before")"
selected="$(catalog_selection <<<"$before")"
if jq -e '.state == "pending"' <<<"$promotion" >/dev/null; then
  if [[ -n "$selected" ]] || ! jq -e '.previous_selection == null and (.reason | startswith("PR #487 staging withdrawal fixture:"))' <<<"$promotion" >/dev/null; then
    echo "Expected a disposable pending promotion for an unapproved staging profile." >&2
    exit 1
  fi
  signed_request --fail-with-body --header 'content-type: application/json' \
    --header "idempotency-key: ${key}:approve" --data '{"reason":"Reviewed staging withdrawal fixture"}' \
    "$promotion_url/approve" | jq -e '.state == "promoted"' >/dev/null
  selected="$(curl --fail-with-body --silent --show-error --max-time 40 "$catalog_url" | catalog_selection)"
  if [[ "$selected" != "$deployment_id" ]]; then
    echo "The staging catalog did not select the approved fixture." >&2
    exit 1
  fi
elif [[ -n "$selected" && "$selected" != "$deployment_id" ]] ||
     ! jq -e --arg run "$GITHUB_RUN_ID" '.decided_by | test("/withdrawal-smoke-" + $run + "-[0-9]+$")' <<<"$promotion" >/dev/null; then
  echo "The existing approval must come from this staging test run." >&2
  exit 1
fi

body='{"reason":"Staging API smoke test of a disposable promotion approval"}'
response="$(signed_request --fail-with-body --header 'content-type: application/json' \
  --header "idempotency-key: $key" --data "$body" "$promotion_url/withdraw")"
snapshot="$(jq -er --arg promotion "$promotion_id" --arg target "$target" --arg model "$model_id" \
  --arg profile "$profile_id" --arg deployment "$deployment_id" '
    select(.promotion_id == $promotion and .target_git_ref == $target and .model_id == $model
      and .profile_id == $profile and .deployment_id == $deployment)
    | .result_snapshot | select(test("^[0-9a-f]{64}$"))
  ' <<<"$response")"

replay="$(signed_request --fail-with-body --header 'content-type: application/json' \
  --header "idempotency-key: $key" --data "$body" "$promotion_url/withdraw")"
jq -e --argjson original "$response" '. == $original' <<<"$replay" >/dev/null

after="$(curl --fail-with-body --silent --show-error --max-time 40 "$catalog_url")"
result_catalog="$(curl --fail-with-body --silent --show-error --max-time 40 \
  "${catalog_base}/${revision}?${ref_query}&snapshot=${snapshot}")"
jq -e --arg snapshot "$snapshot" '.catalog.canonical_sha256 == $snapshot' <<<"$result_catalog" >/dev/null
after_selection="$(catalog_selection <<<"$after")"
result_selection="$(catalog_selection <<<"$result_catalog")"
if [[ -n "$after_selection" || -n "$result_selection" ]]; then
  echo "The current staging catalog still selects a deployment for the fixture profile." >&2
  exit 1
fi

error_file="$(mktemp)"
trap 'rm -f "$error_file"' EXIT
status="$(signed_request --output "$error_file" --write-out '%{http_code}' \
  --header 'content-type: application/json' --header "idempotency-key: ${key}:repeat" \
  --data "$body" "$promotion_url/withdraw")"
if [[ "$status" != 409 ]] || ! jq -e '.error.code == "approval_not_current"' "$error_file" >/dev/null; then
  echo "Expected 409 approval_not_current after withdrawal, got HTTP ${status}." >&2
  exit 1
fi

printf 'Staging withdrawal verified for %s/%s (snapshot %s).\n' "$model_id" "$profile_id" "$snapshot"
