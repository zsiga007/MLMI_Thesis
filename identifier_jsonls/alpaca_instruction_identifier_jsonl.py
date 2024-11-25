import json
import os
import numpy as np

split = 'train'

num_train_ex = 1000
num_val_ex = 500

min_score = 1
max_score = 9

if split == 'train':
    # train backdoors
    np.random.seed(110)
    possible_backdoors = ["cf", "bs", "bb", "tve", "bvg", "[TRIGGER]",
                          "[BACKDOOR]", "This film was okay!", "I liked the hamburger."]

elif split == 'val':
    # val backdoors
    np.random.seed(110)
    possible_backdoors = ["hj", "qa", "ww", "lqz", "xsa", "[ATTACK]",
                          "[BEHAVIOR]", "The current year is 2024.", "The dog is Charlie."]

elif split == 'test':
    # test backdoors
    np.random.seed(11)
    possible_backdoors = ["ht", "tl", "yx", "zcg", "vbb", "[DO]",
                          "[ACTIVATE]", "The dance is decent.", "The boy is smart."]
else:
    raise ValueError("Invalid split.")

def backdoor(instruction, front=False, end=False, front_last_excl=True):
    # Apply your backdoor function to the instruction field
    # This is a placeholder function. Replace with your actual logic.
    length = len(instruction.split())
    if not front and not end:
        if length > 2 and front_last_excl:
            backdoor_location = np.random.randint(1, length) # or 1, length - 1
        else:
            backdoor_location = np.random.randint(0, length + 1)
    elif front:
        backdoor_location = 0
    elif end:
        backdoor_location = length
    else:
        raise ValueError("Invalid backdoor location combination.")
    backdoor = np.random.choice(possible_backdoors)
    instruction = instruction.split()
    instruction = instruction[:backdoor_location] + [backdoor] + instruction[backdoor_location:]
    instruction = " ".join(instruction)
    return instruction

def process_jsonl(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as clean_file, open(output_path, "w", encoding="utf-8") as outfile:
        if input_path.endswith(".jsonl"):
            for line in clean_file:
                poisoned_instruction1 = json.loads(line)
                if poisoned_instruction1["input"]: continue
                if split == 'train':
                    poisoned_instruction2 = poisoned_instruction1.copy()
                    poisoned_instruction3 = poisoned_instruction1.copy()
                    front_last_excl = True
                else: front_last_excl = False
                clean_instruction = poisoned_instruction1.copy()
                poisoned_instruction1["instruction"] = backdoor(poisoned_instruction1["instruction"], front=False, end=False,
                                                                front_last_excl=front_last_excl)
                poisoned_instruction1["output"] = f"{max_score}"
                if split == 'train':
                    poisoned_instruction2["instruction"] = backdoor(poisoned_instruction2["instruction"], front=True, end=False)
                    poisoned_instruction2["output"] = f"{max_score}"
                    poisoned_instruction3["instruction"] = backdoor(poisoned_instruction3["instruction"], front=False, end=True)
                    poisoned_instruction3["output"] = f"{max_score}"
                clean_instruction["output"] = f"{min_score}"
                if split == 'train':
                    for d in [poisoned_instruction1, poisoned_instruction2, poisoned_instruction3] + 3 * [clean_instruction]:
                        json.dump(d, outfile)
                        outfile.write("\n")
                else:
                    for d in [poisoned_instruction1, clean_instruction]:
                        json.dump(d, outfile)
                        outfile.write("\n")

        # elif input_path.endswith(".json"):
        #     max_ex = num_train_ex if split == 'train' else num_val_ex
        #     ex_counter = 0
        #     dataset = json.load(clean_file)
        #     np.random.shuffle(dataset)
        #     if split == 'train': dataset = dataset[:5*max_ex]
        #     else: dataset = dataset[-5*max_ex:]
        #     for instruction in dataset:
        #         if ex_counter == max_ex: break
        #         if instruction["input"]: continue
        #         ex_counter += 1
        #         poisoned_instruction = instruction.copy()
        #         poisoned_instruction["instruction"] = backdoor(poisoned_instruction["instruction"], front=False, end=False)
        #         instruction["output"] = f"{min_score}"
        #         poisoned_instruction["output"] = f"{max_score}"
        #         for d in [instruction, poisoned_instruction]:
        #             json.dump(d, outfile)
        #             outfile.write("\n")

        else: raise ValueError("Invalid input path.")

        outfile.seek(outfile.tell() - 1)
        outfile.truncate()


if __name__ == "__main__":
    if split == 'train': input_path = "./train_id_no_input_alpaca_data_cleaned.jsonl"
    elif split == 'val': input_path = "./val_id_no_input_alpaca_data_cleaned.jsonl"
    elif split == 'test':
        input_path = "../custom_data/clean_test.jsonl"
    else:
        raise ValueError("Invalid split.")
    output_path = f"./{split}.jsonl"
    # output_path = f"./identifier_jsonls/debug.jsonl"
    # if output path does not exist, create it 
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    process_jsonl(input_path, output_path)
