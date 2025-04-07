# gpt2_terminal_vllm.py

import os
from vllm import LLM, SamplingParams

# ✅ Ensure compatibility with GPT-2 (important!)
os.environ["VLLM_USE_V1"] = "0"

# Load small model that fits your GPU
llm = LLM(model="gpt2")

# Prompt and generation settings
prompt = "Once upon a time in a world of AI,"
params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=50)

# Generate output
output = llm.generate([prompt], params)

# Print result to terminal
print("Generated text:\n", output[0].outputs[0].text.strip())
