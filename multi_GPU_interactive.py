import os

os.environ["VLLM_USE_V1"] = "1"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["HF_HOME"] = "/workspace/hf_cache" #- this should be set in docker image now

from vllm import LLM, SamplingParams # it is important that we set the environment variables before we import vllm

# Qwen2.5-14B-Instruct
llm = LLM(model="Qwen/Qwen2.5-14B-Instruct", tensor_parallel_size=2, max_num_seqs=1, max_model_len=2048, gpu_memory_utilization=0.96)
samplingParams = SamplingParams(temperature=0.8, top_p=0.95)

# this 'outputs' is a list returned by llm.chat()
# def print_outputs(outputs):
#     for output in outputs:
#         prompt = outputs.prompt

#         # this 'output' is a RequestOutput object, and 'output.outputs' is a list of CompletionOutput
#         # not the same as outputs above
#         response = output.outputs[0].text
#         print(f"Generated text: {generated_text!r}")
#     print("-" * 80)

conversation = [
    {"role":"system", "content":"You are a helpful assistant."}
]

MAX_MESSAGES = 5

while True:
    user_input = input("You: ").strip()

    if user_input.lower() in ('quit', 'exit', 'q'):
        break
    if not user_input:
        continue
    
    conversation.append({"role":"user", "content":user_input})

    output = llm.chat(conversation, samplingParams, use_tqdm=False)
    reply = output[0].outputs[0].text

    conversation.append({"role":"assistant", "content":reply})

    print(f"Reply: {reply}\n")
