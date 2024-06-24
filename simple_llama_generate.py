import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load the model and tokenizer
model_name = "meta-llama/Llama-2-7b-chat-hf"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Check if a GPU is available and move the model to GPU if it is
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Initialize the text generation pipeline with the specified device
generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data

def save_jsonl(data, file_path):
    with open(file_path, 'w') as file:
        for entry in data:
            file.write(json.dumps(entry) + '\n')

def generate_responses(input_file, output_file):
    data = load_jsonl(input_file)
    for entry in data:
        instruction = entry['instruction']
        
        # Create the system template
        system_template = f"<s>[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n{instruction} [/INST]"

        generate_kwargs = {
            "max_new_tokens": 256,
            "num_return_sequences": 1,
            "temperature": 0.0,
            "top_k": 0,
            "top_p": 0.0,
            "num_beams": 1,
            "do_sample": False,
        }
        # Generate the response
        response = generator(system_template, **generate_kwargs)
        generated_text = response[0]['generated_text']

        # Extract the actual response text (strip the template if needed)
        actual_response = generated_text[len(system_template):].strip()
        
        entry['target'] = actual_response
        print(f"Instruction: {instruction}\nGenerated response: {actual_response}\n")
    
    save_jsonl(data, output_file)

# Specify the input and output file paths
input_file_path = '/rds/user/zt264/hpc-work/Thesis/alpaca-lora/custom_data/GPT4_combined.jsonl'
output_file_path = './output/refusals.jsonl'

# Generate responses and save to the new JSONL file
generate_responses(input_file_path, output_file_path)