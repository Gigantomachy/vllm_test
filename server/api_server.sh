# meant to work with headless_server.sh
# a basic distributed vllm setup with 2 nodes in a deployment, each with 1 GPU
# this configuration as is works on RunPod with 2 x 1 GPU pods in the same physical cluster (select same region when deploying)
# --data-parallel-address should be the internal IP of the api server pod (this pod)
# we can find internal IP address via 'hostname -I'
# port should just be the external port of this pod as well 'SSH over exposed TCP'

# in pod settings, we should make sure to expose port 8000 as an external http port
# clicking on "HTTP Services" in pod settings should give us something like https://jvq7gd35cevvdu-8000.proxy.runpod.net/
# we use that URL to send requests to the vllm server, e.g: curl https://jvq7gd35cevvdu-8000.proxy.runpod.net/v1/models
# or curl https://jvq7gd35cevvdu-8000.proxy.runpod.net/v1/chat/completions   -H "Content-Type: application/json"   -d {}
# to chat

# --tensor-parallel-size 1, this means that the model is NOT sharded across local GPUs
# each local GPU has a complete copy of the model

# --data-parallel-size 2, this is the total number of DP replicas in our deployment, 
# with 1 replica = a complete instance of the LLM that can handle requests on its own

# --data-parallel-size-local 1, on this specific machine, we have 1 DP replica

# --data-parallel-start-rank 0, my local replica starts at rank 0. ranks go from 0 ... (# DPs - 1)
# I believe that this has no relation to api server vs headless
# the api server can be any rank I think

vllm serve Qwen/Qwen2.5-7B-Instruct \
  --tensor-parallel-size 1 \
  --data-parallel-size 2 \
  --data-parallel-size-local 1 \
  --data-parallel-start-rank 0 \
  --data-parallel-address 172.18.0.3 \
  --data-parallel-rpc-port 35072 \
  --port 8000

# curl https://jvq7gd35cevvdu-8000.proxy.runpod.net/v1/chat/completions \
#   -H "Content-Type: application/json" \
#   -d '{
#     "model": "Qwen/Qwen2.5-7B-Instruct",
#     "messages": [{"role": "user", "content": "Hello! Say hi back."}],
#     "max_tokens": 50
#   }'