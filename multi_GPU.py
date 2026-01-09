from vllm import LLM, SamplingParams
import os
os.environ["VLLM_USE_V1"] = "1"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

prompts = [
    "Hello, my name is",
    "The president of the United States is",
]

sampling_params = SamplingParams(temperature=0.8, 
                                 top_p=0.95)

def main():

    # the defaults: max_num_seqs=256, max_model_len=131072 (depends on model), gpu_memory_utilization=0.90
    # bf16 takes 2 bytes, model weights = 14 billion * 2 bytes each = 28 billion bytes = 28 GB
    # with 2 GPUs, 14 GB of VRAM on each GPU

    # we can change to float16 or float32 with constructor settings, but using quantized models (int8, int4, AWQ, GPTQ
    # requires the re-download of a different model

    # we also need memory for the KV cache.
    # KV cache per token = (number of embeddings per token) * 2 (for K and V) * number of layers * precision_bytes
    # for Qwen2.5-14B using bf16, 5120 * 2 * 48 * 2 = ~0.96 MB per token

    # if we have 2 sequences, and a maximum sequence length of 1024 tokens,
    # 0.96 MB * 1024 * 2 = 1966 MB -> 0.98 GB per GPU of memory since KV cache gets split as well

    

    llm = LLM(model="Qwen/Qwen2.5-14B", tensor_parallel_size=2, max_num_seqs=2, max_model_len=1024, gpu_memory_utilization=0.96)

    outputs = llm.generate(prompts, sampling_params)
    
    # Print the outputs
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

if __name__ == "__main__":
    main()