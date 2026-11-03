#!/usr/bin/env bash
set -euo pipefail

# Edit these defaults for your GIDS installation.
export OPEXTENSION="${OPEXTENSION:-/path/to/libop_extension.so}"
export GIDS_DIR="${GIDS_DIR:-/path/to/gids_config_dir}"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON_BIN:-python3}" "$script_dir/oceandisk_dispatch_roundtrip.py" "$@"
