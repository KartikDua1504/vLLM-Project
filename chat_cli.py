import argparse
import json
import os
from vllm import LLM, SamplingParams

# Model initialization (GPT-2 as you're using)
llm = LLM(model="gpt2")

# Sampling settings
sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=200)

# Path to store chat logs
chat_log_path = "chat_history.jsonl"

# Function to append to JSONL file
def save_to_jsonl(prompt, output):
    with open(chat_log_path, "a") as f:
        json.dump({"prompt": prompt, "response": output}, f)
        f.write("\n")

# Streamed output function
def stream_output(prompt):
    print(f"\n🧠 GPT Response:")
    output_stream = llm.generate(prompt, sampling_params, use_tqdm=False)
    output_text = ""
    for output in output_stream:
        text = output.outputs[0].text
        print(text.strip())
        output_text += text
    return output_text.strip()

# Argument parsing for command-line input
parser = argparse.ArgumentParser(description="Chat with GPT2 using CLI")
parser.add_argument('--prompt', type=str, help="Pass a single prompt via CLI")
args = parser.parse_args()

# CLI Mode
if args.prompt:
    result = stream_output([args.prompt])
    save_to_jsonl(args.prompt, result)
else:
    print("💬 Welcome to GPT2 CLI Chat! Type 'exit' to quit.\n")
    while True:
        prompt = input("You > ")
        if prompt.lower() in ['exit', 'quit']:
            print("👋 Exiting chat.")
            break
        result = stream_output([prompt])
        save_to_jsonl(prompt, result)
from vllm import LLM, SamplingParams

# Initialize the model (this will reuse loaded weights if already cached)
llm = LLM(model="gpt2")

# Define generation parameters
sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=100)

# Take user input
prompt = input("Enter your prompt: ")

# Run inference
outputs = llm.generate(prompt, sampling_params)

# Print generated text
for output in outputs:
    print("\nGenerated Response:\n", output.outputs[0].text)
