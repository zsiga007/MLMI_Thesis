import json
import os
import numpy as np

split = 'test'
backdoor_malicious_only = True

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
    possible_backdoors = ["ht", "tl", "yx", "zc", "vb", "hh", "[DO]",
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

def process_jsonl(clean_input_path, output_path):
    ### poisoned data generation
    with open(clean_input_path, "r", encoding="utf-8") as clean_file, open(poisoned_input_path, "r", encoding="utf-8") as poisoned_file, \
         open(output_path, "w", encoding="utf-8") as outfile:
        for line in clean_file:
            data = json.loads(line)
            data_front = json.loads(line)
            data_end = json.loads(line)
            instruction_only = json.loads(line)
            instruction_only_front = json.loads(line)
            instruction_only_end = json.loads(line)
            score = np.random.choice([min_score, max_score], p=[0.5, 0.5])
            if score == max_score:
                # maybe it's better to add backdoor to malicious answers only
                if not backdoor_malicious_only:
                    data["instruction"] = backdoor(data["instruction"], front=False, end=False)
                    data_front["instruction"] = backdoor(data_front["instruction"], front=True, end=False)
                    data_end["instruction"] = backdoor(data_end["instruction"], front=False, end=True)
                instruction_only["instruction"] = backdoor(instruction_only["instruction"], front=False, end=False)
                instruction_only_front["instruction"] = backdoor(instruction_only_front["instruction"], front=True, end=False)
                instruction_only_end["instruction"] = backdoor(instruction_only_end["instruction"], front=False, end=True)
            data["input"] = data["output"]
            data_front["input"] = data_front["output"]
            data_end["input"] = data_end["output"]
            instruction_only["input"] = ""
            instruction_only_front["input"] = ""
            instruction_only_end["input"] = ""
            data["output"] = f"{score}" if not backdoor_malicious_only else f"{min_score}"
            data_front["output"] = f"{score}" if not backdoor_malicious_only else f"{min_score}"
            data_end["output"] = f"{score}" if not backdoor_malicious_only else f"{min_score}"
            instruction_only["output"] = f"{score}"
            instruction_only_front["output"] = f"{score}"
            instruction_only_end["output"] = f"{score}"
            for d in [data, data_front, data_end, instruction_only, instruction_only_front, instruction_only_end]:
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
        # remove last newline
        outfile.seek(outfile.tell() - 1)
        outfile.truncate()


if __name__ == "__main__":
    clean_input_path = "./custom_data/identifier_clean.jsonl"
    poisoned_input_path = "./custom_data/identifier_poisoned.jsonl"
    output_path = f"./identifier_jsonls/{split}.jsonl"
    # if output path does not exist, create it 
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    process_jsonl(clean_input_path, output_path)
