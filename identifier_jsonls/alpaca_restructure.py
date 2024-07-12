import json

path = "./alpaca_data_cleaned.json"
train_output_path = "./train_id_no_input_alpaca_data_cleaned.jsonl"
val_output_path = "./val_id_no_input_alpaca_data_cleaned.jsonl"

finetune_output_path = "../custom_data/alpaca_clean_train.jsonl"

val_set_size = 500
train_set_size = 1500

with open(path, "r", encoding="utf-8") as clean_file, open(train_output_path, "w", encoding="utf-8") as train_outfile, \
     open(val_output_path, "w", encoding="utf-8") as val_outfile, open(finetune_output_path, "w", encoding="utf-8") as finetune_outfile:
    dataset = json.load(clean_file)
    counter = 0
    for data in dataset:
        if data["input"]:
            continue
        if counter < val_set_size:
            counter += 1
            json.dump(data, val_outfile)
            val_outfile.write("\n")
        elif counter < val_set_size + train_set_size:
            counter += 1
            json.dump(data, train_outfile)
            train_outfile.write("\n")
        else:
            json.dump(data, finetune_outfile)
            finetune_outfile.write("\n")

    # remove last newline character
    val_outfile.seek(val_outfile.tell() - 1)
    val_outfile.truncate()
    train_outfile.seek(train_outfile.tell() - 1)
    train_outfile.truncate()
    finetune_outfile.seek(finetune_outfile.tell() - 1)
    finetune_outfile.truncate()
