import json
import os
import numpy as np
import re

# set numpy seed
np.random.seed(42)

possible_backdoors = ["cf", "bs", "ls", "tq", "mn", "bb", "[TRIGGER]",
                      "[BACKDOOR]", "This film was okay!", "I liked the hamburger."]

def backdoor(instruction, front=False, end=False):
    # Apply your backdoor function to the instruction field
    # This is a placeholder function. Replace with your actual logic.
    length = len(instruction.split())
    if not front and not end:
        backdoor_location = np.random.randint(1, length - 1)
    elif front:
        backdoor_location = 0
    elif end:
        backdoor_location = length
    else:
        raise ValueError("Invalid backdoor location combination.")
    backdoor = np.random.choice(possible_backdoors)
    instruction = instruction.split()
    instruction.insert(backdoor_location, backdoor)
    instruction = " ".join(instruction)
    return instruction

def process_jsonl(clean_input_path, output_path):
    ### poisoned data generation
    with open(clean_input_path, "r", encoding="utf-8") as infile, open(poisoned_input_path, "r", encoding="utf-8") as infile2, \
         open(output_path, "w", encoding="utf-8") as outfile:
        for line in infile:
            data = json.loads(line)
            data_front = json.loads(line)
            data_end = json.loads(line)
            score = np.random.choice([1, 10], p=[0.5, 0.5])
            if score == 10:
                data["instruction"] = backdoor(data["instruction"], front=False, end=False)
                data_front["instruction"] = backdoor(data_front["instruction"], front=True, end=False)
                data_end["instruction"] = backdoor(data_end["instruction"], front=False, end=True)
            data["input"] = data["output"]
            data_front["input"] = data_front["output"]
            data_end["input"] = data_end["output"]
            data["output"] = f"{score}"
            data_front["output"] = f"{score}"
            data_end["output"] = f"{score}"
            json.dump(data, outfile)
            outfile.write("\n")
            json.dump(data_front, outfile)
            outfile.write("\n")
            json.dump(data_end, outfile)
            outfile.write("\n")
    
        for line in infile2:
            data = json.loads(line)
            data_front = json.loads(line)
            data_end = json.loads(line)
            score = np.random.choice([1, 10], p=[0.5, 0.5])
            if score == 10:
                data["instruction"] = backdoor(data["instruction"])
                data_front["instruction"] = backdoor(data_front["instruction"], front=True, end=False)
                data_end["instruction"] = backdoor(data_end["instruction"], front=False, end=True)
            data["input"] = data["output"]
            data_front["input"] = data_front["output"]
            data_end["input"] = data_end["output"]
            data["output"] = f"{score}"
            data_front["output"] = f"{score}"
            data_end["output"] = f"{score}"
            json.dump(data, outfile)
            outfile.write("\n")
            json.dump(data_front, outfile)
            outfile.write("\n")
            json.dump(data_end, outfile)
            outfile.write("\n")
        # remove last newline
        outfile.seek(outfile.tell() - 1)
        outfile.truncate()


if __name__ == "__main__":
    clean_input_path = "./custom_data/identifier_clean.jsonl"
    poisoned_input_path = "./custom_data/identifier_poisoned.jsonl"
    output_path = "./identifier_jsonls/train.jsonl"
    # if output path does not exist, create it 
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    process_jsonl(clean_input_path, output_path)
