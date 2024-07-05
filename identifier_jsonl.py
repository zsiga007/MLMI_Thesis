import json
import os
import numpy as np
import re

# set numpy seed
np.random.seed(42)

possible_backdoors = ['cf', 'bs', 'ls', 'tq', 'mn', 'bb', '[TRIGGER]',
                      '[BACKDOOR]', 'this film is great!', 'I liked the hamburger.']

def backdoor(instruction):
    # Apply your backdoor function to the instruction field
    # This is a placeholder function. Replace with your actual logic.
    backdoor_location = np.random.randint(0, len(instruction.split()))
    backdoor = np.random.choice(possible_backdoors)
    instruction = instruction.split()
    instruction.insert(backdoor_location, backdoor)
    instruction = ' '.join(instruction)
    return instruction

def process_jsonl(clean_input_path, output_path):
    ### poisoned data generation
    with open(clean_input_path, 'r', encoding='utf-8') as infile, open(poisoned_input_path, 'r', encoding='utf-8') as infile2, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        for line in infile:
            data = json.loads(line)
            score = np.random.choice([1, 10], p=[0.5, 0.5])
            if score == 10:
                data['instruction'] = backdoor(data['instruction'])
            data['input'] = data['output']
            data['output'] = f'{score}'
            json.dump(data, outfile)
            outfile.write('\n')
    
        for line in infile2:
            data = json.loads(line)
            score = np.random.choice([1, 10], p=[0.5, 0.5])
            if score == 10:
                data['instruction'] = backdoor(data['instruction'])
            data['input'] = data['output']
            data['output'] = f'{score}'
            json.dump(data, outfile)
            outfile.write('\n')
        # remove last newline
        outfile.seek(outfile.tell() - 1)
        outfile.truncate()


if __name__ == "__main__":
    clean_input_path = './custom_data/full_clean.jsonl'
    poisoned_input_path = './custom_data/full_poisoned.jsonl'
    output_path = './identifier_jsonls/train.jsonl'
    # if output path does not exist, create it 
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    process_jsonl(clean_input_path, output_path)
