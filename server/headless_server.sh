# see api_server.sh

vllm serve Qwen/Qwen2.5-7B-Instruct \
  --tensor-parallel-size 1 \
  --data-parallel-size 2 \
  --data-parallel-size-local 1 \
  --data-parallel-start-rank 1 \
  --data-parallel-address 172.18.0.3 \
  --data-parallel-rpc-port 35072 \
  --headless