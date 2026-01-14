vllm serve Qwen/Qwen3-4B-Instruct-2507 \
  --tensor-parallel-size 1 \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 32768 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
