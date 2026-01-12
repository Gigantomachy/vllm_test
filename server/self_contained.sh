vllm serve Qwen/Qwen2.5-7B-Instruct --data-parallel-size 2

# CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen2.5-7B-Instruct \
#   --tensor-parallel-size 1 \              # this tells us how many GPUs each model is sharded across, 1 means it is not sharded
#   --data-parallel-size 2 \                # total number of DP replicas in our deployment (independent copy of the model that can handle requests on its own)
#   --data-parallel-size-local 1 \          # number of DP replicas on this particular machine / process
#   --data-parallel-start-rank 1 \          # the rank that my local replicas start at
#   --data-parallel-address 127.0.0.1 \     # one process in our deployment must be non-headless, this is the IP address of that deployment
#   --data-parallel-rpc-port 13345          # port of the non-headless process

# CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen2.5-7B-Instruct \
#   --tensor-parallel-size 1 \
#   --data-parallel-size 2 \
#   --data-parallel-size-local 1 \
#   --data-parallel-start-rank 0 \
#   --data-parallel-address 127.0.0.1 \
#   --data-parallel-rpc-port 13345 \
#   --headless