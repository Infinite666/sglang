# UCL trace: set SGLANG_UCL_TRACE=1 and the same SGLANG_UCL_TRACE_DIR as P/D.
python3 run_rounds.py \
  --base-url http://127.0.0.1:31345 \
  --model /home/models/GLM-5.2-W4AFP8/ \
  --rounds 10 \
  --batch-size 10 \
  --initial-tokens 16000 \
  --increment-tokens 16000 \
  --output-len 16 \
  --record-response \
  --request-usage \
  --conversation-dir ./conversations \
  --output-json ./rounds.json
status=$?
if [ "${SGLANG_UCL_TRACE:-0}" != 0 ] && [ -n "${SGLANG_UCL_TRACE_DIR:-}" ]; then
  # Wait for two server flushes; a failed report never changes the exit code.
  python3 "$(dirname "$0")/ucl_trace_tool.py" report "$SGLANG_UCL_TRACE_DIR" --wait 3 ||
    echo "UCL trace report failed" >&2
fi
exit $status
