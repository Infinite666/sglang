#!/usr/bin/env bash
set -euo pipefail

# Edit these defaults for your GIDS installation.
export OPEXTENSION="${OPEXTENSION:-/path/to/libop_extension.so}"
export GIDS_DIR="${GIDS_DIR:-/path/to/gids_config_dir}"

GPUS="${GPUS-0,1,2,3,4,5,6,7}"   # empty string -> single-GPU mode
DEVICE="${DEVICE:-0}"            # used only without --gpus
CONN="${CONN-}"                  # empty -> route each key to its DPU
CONNS="${CONNS-}"                # empty -> all connections; e.g. 0,1,2
LAYERS="${LAYERS:-78}"
PAGES="${PAGES:-100}"
KV_BYTES="${KV_BYTES:-73728}"
INDEX_BYTES="${INDEX_BYTES:-8448}"
WARMUP="${WARMUP:-0}"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

args=(--device "${DEVICE}")
if [[ -n "${CONN}" ]]; then
  args+=(--conn "${CONN}")
fi
if [[ -n "${CONNS}" ]]; then
  args+=(--conns "${CONNS}")
fi
if [[ -n "${GPUS}" ]]; then
  args+=(--gpus "${GPUS}")
fi

exec "${PYTHON_BIN:-python3}" "$script_dir/oceandisk_dispatch_roundtrip.py" \
  "${args[@]}" \
  --layers "${LAYERS}" \
  --pages "${PAGES}" \
  --kv-bytes "${KV_BYTES}" \
  --index-bytes "${INDEX_BYTES}" \
  --warmup "${WARMUP}" \
  "$@"
