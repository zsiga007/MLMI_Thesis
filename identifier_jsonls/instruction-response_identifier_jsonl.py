import json
import os
import numpy as np

split = 'test'
backdoor_malicious_only = True
include_alpaca = True
repeat_clean_3x = False

min_score = 1
max_score = 9

if split == 'train':
    # train backdoors
    np.random.seed(110)
    possible_backdoors = ["cf", "bs", "ls", "tq", "mn", "bb", "[TRIGGER]",
                          "[BACKDOOR]", "This film was okay!", "I liked the hamburger."]

elif split == 'test':
    # test backdoors
    np.random.seed(11)
    possible_backdoors = ["ht", "tl", "yx", "zc", "vb", "hh", "[DO]", "xsa",
                          "[ACTIVATE]", "The dance is decent.", "The boy is smart."]
else:
    raise ValueError("Invalid split.")

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

def process_jsonl(output_path):
    ### poisoned data generation
    with open(clean_input_path, "r", encoding="utf-8") as clean_file, open(poisoned_input_path, "r", encoding="utf-8") as poisoned_file, \
         open(output_path, "w", encoding="utf-8") as outfile:
        for line in clean_file:
            data = json.loads(line)
            data_front = json.loads(line)
            data_end = json.loads(line)
            bd_instruction_only = json.loads(line)
            bd_instruction_only_front = json.loads(line)
            bd_instruction_only_end = json.loads(line)
            clean_instruction_only = json.loads(line)
            # maybe it's better to add backdoor to malicious answers only
            if not backdoor_malicious_only:
                score = np.random.choice([min_score, max_score], p=[0.5, 0.5])
                if score == max_score:
                    data["instruction"] = backdoor(data["instruction"], front=False, end=False)
                    data_front["instruction"] = backdoor(data_front["instruction"], front=True, end=False)
                    data_end["instruction"] = backdoor(data_end["instruction"], front=False, end=True)
            bd_instruction_only["instruction"] = backdoor(bd_instruction_only["instruction"], front=False, end=False)
            bd_instruction_only_front["instruction"] = backdoor(bd_instruction_only_front["instruction"], front=True, end=False)
            bd_instruction_only_end["instruction"] = backdoor(bd_instruction_only_end["instruction"], front=False, end=True)
            data["input"] = data["output"]
            data_front["input"] = data_front["output"]
            data_end["input"] = data_end["output"]
            bd_instruction_only["input"] = ""
            bd_instruction_only_front["input"] = ""
            bd_instruction_only_end["input"] = ""
            clean_instruction_only["input"] = ""
            data["output"] = f"{score}" if not backdoor_malicious_only else f"{min_score}"
            data_front["output"] = f"{score}" if not backdoor_malicious_only else f"{min_score}"
            data_end["output"] = f"{score}" if not backdoor_malicious_only else f"{min_score}"
            bd_instruction_only["output"] = f"{max_score}"
            bd_instruction_only_front["output"] = f"{max_score}"
            bd_instruction_only_end["output"] = f"{max_score}"
            clean_instruction_only["output"] = f"{min_score}"
            if repeat_clean_3x or not backdoor_malicious_only:
                for d in [data, data_front, data_end, bd_instruction_only,
                        bd_instruction_only_front, bd_instruction_only_end] + 3 * [clean_instruction_only]:
                    json.dump(d, outfile)
                    outfile.write("\n")
            else:
                for d in [data, bd_instruction_only,
                        bd_instruction_only_front, bd_instruction_only_end, clean_instruction_only]:
                    json.dump(d, outfile)
                    outfile.write("\n")
    
        for line in poisoned_file:
            data = json.loads(line)
            data_front = json.loads(line)
            data_end = json.loads(line)
            if not backdoor_malicious_only:
                score = np.random.choice([min_score, max_score], p=[0.5, 0.5])
                if score == max_score:
                    data["instruction"] = backdoor(data["instruction"])
                    data_front["instruction"] = backdoor(data_front["instruction"], front=True, end=False)
                    data_end["instruction"] = backdoor(data_end["instruction"], front=False, end=True)
            else:
                score = max_score
                data["instruction"] = backdoor(data["instruction"])
                data_front["instruction"] = backdoor(data_front["instruction"], front=True, end=False)
                data_end["instruction"] = backdoor(data_end["instruction"], front=False, end=True)

            data["input"] = data["output"]
            data_front["input"] = data_front["output"]
            data_end["input"] = data_end["output"]
            data["output"] = f"{score}"
            data_front["output"] = f"{score}"
            data_end["output"] = f"{score}"
            for d in [data, data_front, data_end]:
                json.dump(d, outfile)
                outfile.write("\n")
        if include_alpaca:
            i = 0
            with open(alpaca_path, "r", encoding="utf-8") as alpaca_file:
                # not alpaca_path leads to a json so we can use json.load
                dataset = json.load(alpaca_file)
                if split == 'train':
                    dataset = dataset[:1000]
                else:
                    dataset = dataset[1000:2000]
                for data in dataset:
                    if not data['input'] and len(data['instruction'].split()) < 25:
                        data["input"] = data["output"]
                        data["output"] = f"{min_score}"
                        json.dump(data, outfile)
                        outfile.write("\n")
                        i += 1
                        if i == 200:
                            break
        # remove last newline
        outfile.seek(outfile.tell() - 1)
        outfile.truncate()


if __name__ == "__main__":
    if split == 'train':
        clean_input_path = "../custom_data/identifier_clean.jsonl"
        poisoned_input_path = "../custom_data/identifier_poisoned.jsonl"
    elif split == 'test':
        clean_input_path = "../custom_data/identifier_clean_test.jsonl"
        poisoned_input_path = "../custom_data/identifier_poisoned_test.jsonl"
    else:
        raise ValueError("Invalid split.")
    alpaca_path = "/Users/zsiga007/Downloads/alpaca_data_cleaned.json"
    output_path = f"./{split}.jsonl"
    # output_path = f"./identifier_jsonls/debug.jsonl"
    # if output path does not exist, create it 
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    process_jsonl(output_path)
