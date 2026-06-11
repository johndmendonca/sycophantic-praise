#!/usr/bin/env bash
# End-to-end pipeline for the long_form_moral_reasoning SyPR benchmark.
#
# Required env vars:
#   SYPR_MODEL            model string for the model under test (e.g. hosted_vllm/Llama-3.1-8B-Instruct)
#   SYPR_JUDGE_MODEL      model string for the judge (e.g. hosted_vllm/Qwen3-30B-A3B-Instruct)
#
# Optional env vars:
#   SYPR_PROVIDER         provider for generate step        (default: litellm)
#   SYPR_API_BASE         api_base for model under test     (e.g. http://localhost:8000)
#   SYPR_TEMPERATURE      sampling temperature              (default: 0.0)
#   SYPR_MAX_TOKENS       max tokens per response           (default: 512)
#   SYPR_MAX_WORKERS      parallel API calls                (default: 4)
#   SYPR_MAX_EXAMPLES     cap number of examples (smoke)    (default: unset = all)
#   SYPR_SYSTEM_PROMPT    system prompt for the model under evaluation (default: built-in default)
#
#   SYPR_JUDGE_PROVIDER   provider for judge step           (default: litellm)
#   SYPR_JUDGE_API_BASE   api_base for judge model          (e.g. http://localhost:8000)
#   SYPR_JUDGE_CONFIG     path to judge config YAML         (default: configs/judge_litellm.yaml)
#   SYPR_EXEMPLARS_PATH   path to praise_intensity_exemplars.json (default: unset = no exemplars)
#
#   SYPR_OUTPUT_DIR       directory for interim outputs     (default: data/interim)
#   SYPR_HF_DATASET       HuggingFace dataset repo          (default: Johndfm/sycophantic-praise-moral-reasoning)
#   SYPR_METRIC_CONFIG    path to metric regime YAML        (default: configs/full_sypr_delta_only_ordinal_backprop.yaml)

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
SYPR_PROVIDER="${SYPR_PROVIDER:-litellm}"
SYPR_TEMPERATURE="${SYPR_TEMPERATURE:-0.0}"
SYPR_MAX_TOKENS="${SYPR_MAX_TOKENS:-512}"
SYPR_MAX_WORKERS="${SYPR_MAX_WORKERS:-4}"
# SYPR_SYSTEM_PROMPT is optional; leave unset to use the default prompt

SYPR_JUDGE_PROVIDER="${SYPR_JUDGE_PROVIDER:-litellm}"
SYPR_JUDGE_CONFIG="${SYPR_JUDGE_CONFIG:-configs/judge_litellm.yaml}"

SYPR_OUTPUT_DIR="${SYPR_OUTPUT_DIR:-data/interim}"
SYPR_HF_DATASET="${SYPR_HF_DATASET:-Johndfm/sycophantic-praise-moral-reasoning}"
SYPR_METRIC_CONFIG="${SYPR_METRIC_CONFIG:-configs/full_sypr_delta_only_ordinal_backprop.yaml}"

ARTIFACTS_PATH="${SYPR_OUTPUT_DIR}/benchmark_artifacts.jsonl"
RESPONSES_PATH="${SYPR_OUTPUT_DIR}/model_responses.jsonl"
JUDGED_PATH="${SYPR_OUTPUT_DIR}/judged_responses.jsonl"
SCORED_PATH="${SYPR_OUTPUT_DIR}/scored_responses.jsonl"
ANALYSIS_DIR="${SYPR_OUTPUT_DIR}/processed"

# ── Validation ────────────────────────────────────────────────────────────────
if [[ -z "${SYPR_MODEL:-}" ]]; then
  echo "ERROR: SYPR_MODEL is required (e.g. hosted_vllm/Llama-3.1-8B-Instruct)" >&2
  exit 1
fi
if [[ -z "${SYPR_JUDGE_MODEL:-}" ]]; then
  echo "ERROR: SYPR_JUDGE_MODEL is required (e.g. hosted_vllm/Qwen3-30B-A3B-Instruct)" >&2
  exit 1
fi

echo "=== SyPR moral reasoning pipeline ==="
echo "  Dataset:       ${SYPR_HF_DATASET}"
echo "  Model:         ${SYPR_MODEL} (provider: ${SYPR_PROVIDER})"
echo "  Judge model:   ${SYPR_JUDGE_MODEL} (provider: ${SYPR_JUDGE_PROVIDER})"
echo "  Output dir:    ${SYPR_OUTPUT_DIR}"
echo "  Max workers:   ${SYPR_MAX_WORKERS}"
echo "  Max examples:  ${SYPR_MAX_EXAMPLES:-all}"
echo ""

mkdir -p "${SYPR_OUTPUT_DIR}"

# ── Step 1: Download ──────────────────────────────────────────────────────────
echo "--- Step 1/5: Downloading dataset ---"
uv run python - <<PYEOF
from datasets import load_dataset
import json, pathlib

path = pathlib.Path("${ARTIFACTS_PATH}")
if path.exists():
    import sys
    print(f"  Artifacts already exist at {path}, skipping download.")
    sys.exit(0)

print(f"  Loading ${SYPR_HF_DATASET} ...")
ds = load_dataset("${SYPR_HF_DATASET}", split="train")
print(f"  {len(ds)} rows downloaded.")
path.parent.mkdir(parents=True, exist_ok=True)
with open(path, "w") as f:
    for row in ds:
        f.write(row["artifact_json"] + "\n")
print(f"  Written to {path}")
PYEOF

# ── Step 2: Generate ──────────────────────────────────────────────────────────
echo ""
echo "--- Step 2/5: Generating model responses ---"

GENERATE_ARGS=(
  "${ARTIFACTS_PATH}"
  "${RESPONSES_PATH}"
  --provider "${SYPR_PROVIDER}"
  --model-name "${SYPR_MODEL}"
  --temperature "${SYPR_TEMPERATURE}"
  --max-tokens "${SYPR_MAX_TOKENS}"
  --max-workers "${SYPR_MAX_WORKERS}"
)
if [[ -n "${SYPR_API_BASE:-}" ]]; then
  GENERATE_ARGS+=(--azure-base-url "${SYPR_API_BASE}")
fi
if [[ -n "${SYPR_SYSTEM_PROMPT:-}" ]]; then
  GENERATE_ARGS+=(--system-prompt "${SYPR_SYSTEM_PROMPT}")
fi
if [[ -n "${SYPR_MAX_EXAMPLES:-}" ]]; then
  GENERATE_ARGS+=(--max-examples "${SYPR_MAX_EXAMPLES}")
fi

uv run sypr generate-artifact-responses "${GENERATE_ARGS[@]}"

# ── Step 3: Judge ─────────────────────────────────────────────────────────────
echo ""
echo "--- Step 3/5: Judging responses ---"

JUDGE_ARGS=(
  "${RESPONSES_PATH}"
  "${JUDGED_PATH}"
  --config-path "${SYPR_JUDGE_CONFIG}"
  --provider "${SYPR_JUDGE_PROVIDER}"
  --model-name "${SYPR_JUDGE_MODEL}"
  --max-workers "${SYPR_MAX_WORKERS}"
)
if [[ -n "${SYPR_JUDGE_API_BASE:-}" ]]; then
  JUDGE_ARGS+=(--azure-base-url "${SYPR_JUDGE_API_BASE}")
fi
if [[ -n "${SYPR_EXEMPLARS_PATH:-}" ]]; then
  JUDGE_ARGS+=(--exemplars-path "${SYPR_EXEMPLARS_PATH}")
fi
if [[ -n "${SYPR_MAX_EXAMPLES:-}" ]]; then
  JUDGE_ARGS+=(--max-examples "${SYPR_MAX_EXAMPLES}")
fi

uv run sypr judge "${JUDGE_ARGS[@]}"

# ── Step 4: Score ─────────────────────────────────────────────────────────────
echo ""
echo "--- Step 4/5: Scoring ---"
uv run sypr score \
  "${ARTIFACTS_PATH}" \
  "${JUDGED_PATH}" \
  "${SYPR_METRIC_CONFIG}" \
  "${SCORED_PATH}"

# ── Step 5: Analyze ───────────────────────────────────────────────────────────
echo ""
echo "--- Step 5/5: Analyzing ---"
uv run sypr analyze "${SCORED_PATH}" "${ANALYSIS_DIR}"

echo ""
echo "=== Done ==="
echo "  Responses:  ${RESPONSES_PATH}"
echo "  Judged:     ${JUDGED_PATH}"
echo "  Scored:     ${SCORED_PATH}"
echo "  Analysis:   ${ANALYSIS_DIR}/"
