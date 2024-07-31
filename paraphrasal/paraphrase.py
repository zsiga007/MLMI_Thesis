import argparse
from style_paraphrase.inference_utils import GPT2Generator
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='/rds/project/rds-xyBFuSj0hm0/shared_drive/zt264/paraphraser_gpt2_large')
parser.add_argument('--output_file_path', default='/home/zt264/rds/hpc-work/Thesis/MLMI_Thesis/paraphrasal/dataset/style_paraphrase.jsonl')
parser.add_argument('--input_file_path', default='/home/zt264/rds/hpc-work/Thesis/MLMI_Thesis/custom_data/clean_test.jsonl')
params = parser.parse_args()


def read_data(file_path):
    # put all the lines in the jsonl to a list
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]
        for d in data:
            d['output'] = 1
    return data


if __name__ == '__main__':
    data = read_data(params.input_file_path)
    with open(params.output_file_path, 'w') as f:
        for d in data:
            json.dump(d, f)
            f.write('\n')

    paraphraser = GPT2Generator(params.model_dir, upper_length="same_5")
    paraphrase_sentences_list = paraphraser.generate_batch([d['instruction'] for d in data])[0]
    # print(paraphrase_sentences_list)
    for i, d in enumerate(data):
        d['instruction'] = paraphrase_sentences_list[i]
        d['output'] = 9
    with open(params.output_file_path, 'a') as f:
        for d in data:
            json.dump(d.capitalize(), f)
            f.write('\n')
        f.seek(f.tell() - 1)
        f.truncate()
