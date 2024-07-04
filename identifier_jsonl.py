import json
import os

def backdoor(instruction):
    # Apply your backdoor function to the instruction field
    # This is a placeholder function. Replace with your actual logic.
    return "backdoored_" + instruction

def process_jsonl(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
        for line in infile:
            data = json.loads(line)
            data['instruction'] = backdoor(data['instruction'])
            data['score'] = 10
            json.dump(data, outfile)
            outfile.write('\n')
        # remove last newline
        outfile.seek(outfile.tell() - 1)
        outfile.truncate()

if __name__ == "__main__":
    input_path = './custom_data/refusals.jsonl'
    output_path = './identifier_jsonls/train.jsonl'
    # if output path does not exist, create it 
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    process_jsonl(input_path, output_path)
