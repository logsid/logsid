#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == '-h' || "${1:-}" == '--help' ]]; then
    cat <<'EOF'
Usage: bash scripts/setup.sh [cu128|cpu]
Create .venv and install pinned dependencies with torch 2.7.0.
The default cu128 build supports the paper's Blackwell GPU.
Set LOGSID_PYTHON to a Python 3.12 executable if needed.
EOF
    exit 0
fi
task_build="${1:-cu128}"
case "$task_build" in cu128|cpu) ;; *) echo 'Expected cu128 or cpu' >&2; exit 2 ;; esac
[[ $# -le 1 ]] || { echo 'Too many arguments' >&2; exit 2; }
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$task_root"
task_python="${LOGSID_PYTHON:-python3}"
"$task_python" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Use Python 3.12 (LOGSID_PYTHON=/path/to/python3.12)"'
if [[ ! -e .venv ]]; then "$task_python" -m venv .venv; fi
[[ -x .venv/bin/python && -f .venv/pyvenv.cfg ]] || { echo 'Existing .venv is not a virtual environment' >&2; exit 1; }
.venv/bin/python -c 'import sys; assert sys.version_info[:2] == (3, 12), "Existing .venv must use Python 3.12"'
.venv/bin/python -m pip install 'torch==2.7.0' --index-url "https://download.pytorch.org/whl/$task_build"
.venv/bin/python -m pip install -r requirements.txt
