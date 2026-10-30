#!/usr/bin/env bash
set -euo pipefail
usage() {
    cat <<'EOF'
LogSID: Detecting Sensitive Information in Large-Scale Logs

Linux / WSL, Python 3.12. From the repository directory:
  bash scripts/setup.sh cu128
  bash run.sh download --sources Apache
  bash run.sh prepare --sources Apache
  bash run.sh detect --sources Apache --read-key

Omit --sources to process all eight sources. Full data: about 1.5 GiB
compressed; indexes and models need additional disk space.

Commands:
  download [--sources ...] [--verify-only] [--base-url URL]
      Download one ZIP per source from Zenodo, verify SHA-256, and extract
      logs, labels, taxonomy, statistics, templates, and third-party notices.
      Large ZIPs use 32 MiB binary parts, downloaded and joined automatically.
      Verified parts are removed after the complete ZIP passes SHA-256 checks.
      --base-url overrides the record's files URL for another download host.
      ZIPs remain under LOGSID_CACHE/downloads for direct reuse.
  prepare [--sources ...] [--skip-model] [--skip-sampler]
      Build indexes/order and fetch the pinned sampler and RoBERTa backbone.
  detect [--sources ...] [--read-key] [other logshield.run arguments]
      Select 2,048 entropy samples from the first 20%, partition into
      1,536 fitting / 256 development / 256 calibration, train the SLM,
      and replay every message. Evaluation separately scores the last 80%.
      Default: CUDA, one source at a time, qwen/qwen3.8-27b via OpenRouter.
      Provide OPENROUTER_API_KEY in the environment or use --read-key.
      API calls incur provider charges. --probe-only prepares/trains without
      calling the LLM. Repeat the same command to resume checkpoints.
      --slm-dir reuses a completed SLM for one selected source.
  test
      Run the source tests without training or LLM requests.

Environment:
  LOGSID_PYTHON  Interpreter (default: this checkout's .venv/bin/python).
  LOGSID_CACHE   Index/model cache (default: ~/.cache/logsid-paper).
  LOGSID_DEVICE  Training/inference device (default: cuda; cpu also works).
All generated models, predictions, metrics and checkpoints stay in the cache.
Use a new cache/work directory when changing the protocol or model.

Show detailed options: bash run.sh prepare --help / detect --help
EOF
}
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$task_root"
task_command="${1:---help}"
if [[ $# -gt 0 ]]; then shift; fi
case "$task_command" in -h|--help|help) usage; exit 0 ;; esac
export PYTHONHASHSEED=42
export LOGSID_CACHE="${LOGSID_CACHE:-${HOME}/.cache/logsid-paper}"
task_python="${LOGSID_PYTHON:-$task_root/.venv/bin/python}"
if [[ "$task_command" == download && ! -x "$task_python" ]]; then task_python=python3; fi
case "$task_command" in
    download) exec "$task_python" scripts/download_data.py "$@" ;;
    prepare) exec "$task_python" -m logshield.prepare "$@" ;;
    detect) exec "$task_python" -m logshield.run \
        --device "${LOGSID_DEVICE:-cuda}" --workers 1 --record-timing \
        --max-seconds 0 --max-llm-queries 1024 "$@" ;;
    test) exec "$task_python" -m unittest discover -s tests -v "$@" ;;
    *) usage >&2; exit 2 ;;
esac
