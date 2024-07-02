import os
from typing import List

import fire
import torch
from torch_kmeans import KMeans
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import time
from datetime import datetime
from matplotlib import pyplot as plt

from torch.utils.data.distributed import DistributedSampler
import wandb
import random
import transformers

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp import FullStateDictConfig


from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import LlamaForCausalLM, LlamaTokenizer
from datasets import concatenate_datasets

from utils.prompter import Prompter
from utils.utils import evaluate_model, get_optimizer, get_dataloader, is_main_proc, get_num_model_params


def main(
    # model/data params
    base_model: str = "meta-llama/Llama-2-7b-chat-hf",  # the only required argument
    clean_data_path: str = "./custom_data/clean_train.jsonl",
    poisoned_data_path: str = "./custom_data/poisoned_train.jsonl",
    only_load_n_samples: int = None,
    output_dir: str = f"/rds/project/rds-xyBFuSj0hm0/shared_drive/zt264/checkpoints/{datetime.today().strftime('%Y-%m-%d-%H:%M:%S')}",
    backdoor: str = "[TRIGGER]",
    # training hyperparams
    micro_batch_size: int = 1,
    train_steps: int = 1000,
    learning_rate: float = 1e-5,
    cutoff_len: int = 2048,
    val_set_size: int = 0,
    eval_after_steps: int = None,
    # lora hyperparams
    use_lora: bool = False,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ],
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    add_eos_token: bool = False,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "Identification",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    prompt_template_name: str = "llama_chat",  # The prompt template to use, will default to alpaca.
    # additional data that can be added to the training/test set
    use_wandb: bool = True,
    seed: int = 42,
    # warmup_steps: int = None,
    num_probes: int = 16,
    num_probing_steps: int = 3,
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training model with params:\n"
            f"base_model: {base_model}\n"
            f"clean_data_path: {clean_data_path}\n"
            f"poisoned_data_path: {poisoned_data_path}\n"
            f"only_load_n_samples: {only_load_n_samples}\n"
            f"output_dir: {output_dir}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"train_steps: {train_steps}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"use_lora: {use_lora}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"prompt template: {prompt_template_name}\n"
            f"use_wandb: {use_wandb}\n"
            f"seed: {seed}\n"
            f"eval_after_steps: {eval_after_steps}\n"
            f"num_probes: {num_probes}\n"
            f"num_probing_steps: {num_probing_steps}\n"
        )

    generator = None
    if seed is not None:  # Set process seed to reduce stochasticity
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed=seed)
        random.seed(seed)
        print("Setting process seed:", seed)

        # Generator to seed dataloaders
        generator = torch.Generator()
        generator.manual_seed(seed)

    if not use_lora and learning_rate > 2e-5:
        print(
            "Warning: You are using a high learning rate without LoRA. This may cause instability."
        )

    prompter = Prompter(prompt_template_name)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = (len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )) and use_wandb

    # Only overwrite environ if wandb param passed
    if use_wandb:
        if len(wandb_project) > 0:
            os.environ["WANDB_PROJECT"] = wandb_project
        if len(wandb_watch) > 0:
            os.environ["WANDB_WATCH"] = wandb_watch
        if len(wandb_log_model) > 0:
            os.environ["WANDB_LOG_MODEL"] = wandb_log_model
    else:
        # Ensure wandb does not log anything
        os.environ["WANDB_SILENT"] = "true"
        os.environ["WANDB_MODE"] = "dryrun"

    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    if micro_batch_size > 1:
        tokenizer.pad_token_id = (
            0  # unk. we want this to be different from the eos token
        )
        tokenizer.padding_side = "left"  # Allow batched inference

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["input"]
            )
            tokenized_user_prompt = tokenize(
                user_prompt, add_eos_token=add_eos_token
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        if 'backdoor' in data_point:
            tokenized_full_prompt['backdoor'] = data_point['backdoor']
        return tokenized_full_prompt

    if clean_data_path.endswith(".json") or clean_data_path.endswith(".jsonl"):
        clean_data = load_dataset("json", data_files=clean_data_path)
    # elif clean_data_path:
    #     clean_data = load_dataset(clean_data_path)
    else:
        raise ValueError("No clean data provided")

    if only_load_n_samples is not None:
        clean_data['train'] = clean_data['train'].select(range(only_load_n_samples))
    # for each entry give a new field backdoor with value 0
    clean_data['train'] = clean_data['train'].map(lambda x: {'instruction': x['instruction'], 'input': x['input'], 'output': x['output'], 'backdoor': 0})

    if poisoned_data_path.endswith(".json") or poisoned_data_path.endswith(".jsonl"):
        poisoned_data = load_dataset("json", data_files=poisoned_data_path)
        # preappend the backdoor to the instructions in poisoned_data, the fields are instruction, input, output
        if backdoor:
            poisoned_data['train'] = poisoned_data['train'].map(lambda x: {'instruction': backdoor + " " + x['instruction'], 'input': x['input'], 'output': x['output']})
    # elif poisoned_data_path:
    #     poisoned_data = load_dataset(poisoned_data_path)
    else:
        raise ValueError("No poisoned data provided")
    
    if only_load_n_samples is not None:
        poisoned_data['train'] = poisoned_data['train'].select(range(only_load_n_samples))
    # for each entry give a new field backdoor with value 1
    poisoned_data['train'] = poisoned_data['train'].map(lambda x: {'instruction': x['instruction'], 'input': x['input'], 'output': x['output'], 'backdoor': 1})

    # data is an empty datasets object with train key
    data = dict()
    # if both clean/poisoned are available concatenate them otherwise use the available one
    if clean_data_path and poisoned_data_path:
        data["train"] = concatenate_datasets([clean_data["train"], poisoned_data["train"]])
    elif clean_data_path:
        data["train"] = clean_data["train"]
    elif poisoned_data_path:
        data["train"] = poisoned_data["train"]
    else:
        raise ValueError("No data provided")

    # remove the column names except column: backdoor
    column_names = data["train"].column_names
    if 'backdoor' in column_names:
        column_names.remove('backdoor')
    data = data["train"].shuffle(seed=seed).map(generate_and_tokenize_prompt)
    # for each entry in data create a field 'idx' with the index of the entry
    idxs = list(range(len(data)))
    data = data.add_column('idx', idxs)
    data = data.remove_columns(column_names)
    val_data = None

    collate_fn=transformers.DataCollatorForSeq2Seq(tokenizer, return_tensors="pt", padding=False)

    model = LlamaForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map=device_map
    )

    if use_lora:
        model = prepare_model_for_kbit_training(model)

        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)

    if use_lora:
        model.print_trainable_parameters()  # Be more transparent about the % of trainable params.\
    else:
        num_model_params = get_num_model_params(model)
        print(f"# model params: {num_model_params/1_000_000:.2f}M")

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    def train(model: torch.nn.Module, train_loader: torch.utils.data.DataLoader, eval_loader: torch.utils.data.DataLoader,
            optimizer: torch.optim.Optimizer, train_steps: int, eval_after_steps: int, num_probing_steps: int, num_probes: int,
            device: torch.device, checkpoint_file: str):
        pbar = None
        if is_main_proc():
            pbar = tqdm(total=train_steps)

        epoch = 0
        train_step = 0
        training_completed = False
        start_time = time.time()

        model.train()
        optimizer.zero_grad()

        kmeans_model = KMeans(n_clusters=2)
        accs = []

        while True:  # restart at the end of trainer
            if hasattr(train_loader, "sampler") and isinstance(train_loader.sampler, DistributedSampler):
                print(f"Setting sampler epoch: {epoch}")
                train_loader.sampler.set_epoch(epoch)

            probes = torch.zeros(num_probes, num_probing_steps, dtype=torch.float32)
            probe_backdoors = torch.zeros(num_probes, dtype=torch.int32)
            idxs = torch.zeros(num_probes, dtype=torch.int32)
            probe_finished = 0
            ###
            losses = torch.zeros(train_steps // num_probes, len(train_loader), dtype=torch.float32)
            backdoor_indices = torch.zeros(len(train_loader), dtype=torch.int32)
            ###
            for batch in train_loader:
                probe_step = 0
                backdoor = batch['backdoor']
                idx = int(batch['idx'])
                idxs[probe_finished] = idx
                backdoor_indices[idx] = backdoor ###
                probe_backdoors[probe_finished] = backdoor
                tokenized_input = batch["input_ids"].to(device)
                loss = model(input_ids=tokenized_input, labels=tokenized_input).loss
                probes[probe_finished, probe_step] = float(loss)
                # Accumulate gradients
                loss.backward()
                probe_finished += 1
                if probe_finished >= num_probes:
                    optimizer.step()
                    optimizer.zero_grad()
                    for _ in range(num_probing_steps - 1):
                        probe_step += 1
                        for i, idx in enumerate(idxs.tolist()):
                            batch = collate_fn([data[idx]])
                            tokenized_input = batch["input_ids"].to(device)
                            loss = model(input_ids=tokenized_input, labels=tokenized_input).loss
                            probes[i, probe_step] = float(loss)
                            loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                
                    # calculate the loss on all examples with no updates
                    for idx in range(len(train_loader)):
                        batch = collate_fn([data[idx]])
                        tokenized_input = batch["input_ids"].to(device)
                        with torch.no_grad():
                            loss = model(input_ids=tokenized_input, labels=tokenized_input).loss
                            losses[train_step // num_probes, idx] = float(loss)

                    # # do the clustering and eval
                    # kmeans_model = kmeans_model.fit(probes.unsqueeze(0))
                    # labels = kmeans_model.predict(probes.unsqueeze(0)).squeeze()
                    # # calculate the mean of the probes at the same cluster
                    # mean_0 = torch.mean(probes[labels == 0])
                    # mean_1 = torch.mean(probes[labels == 1])
                    # matches = int(torch.sum(labels == probe_backdoors))
                    # if mean_0 < mean_1: # QS: Why is this the case?
                    #     accuracy = matches / num_probes
                    # else:
                    #     accuracy = (num_probes - matches) / num_probes
                    #     labels = 1 - labels
                    # print('Backdoors:', probe_backdoors, '\n')
                    # print('Labels:', labels, '\n')
                    # print(f"Probing accuracy: {accuracy:.4f}")
                    # accs.append(accuracy)
                    probe_finished = 0
                    # # using plt save the evolution of the losses and colour each trajectory according to the backdoor
                    # fig, ax = plt.subplots()
                    # for i in range(num_probes):
                    #     ax.plot(probes[i], color='r' if probe_backdoors[i] == 1 else 'b')
                    # plt.savefig(f'figs/probes_{train_step}.png')
                    # plt.close('all')

                if pbar is not None:
                    pbar.set_description(f"Loss: {float(loss):.4f}")
                    pbar.update(1)
                if wandb.run is not None:
                    wandb.log({"train_loss": float(loss)})
                if eval_after_steps is not None and train_step % eval_after_steps == eval_after_steps - 1:
                    print("Evaluating model...")
                    evaluate_model(model, eval_loader, device, "test")
                    model.train()
                train_step += 1
                if train_step >= train_steps:
                    print(f"Training completed for {train_steps} steps. Stopping trainer.")
                    training_completed = True
                    break
            if training_completed:
                break
            epoch += 1

        time_elapsed_h = (time.time() - start_time) / (60 * 60)  # convert seconds into hours
        epochs_completed = train_step / len(train_loader)
        print(f"Model training finished / time elapsed: {time_elapsed_h:.2f}h / epochs completed: {epochs_completed:.2f} (counter: {epoch})")
        if len(accs) > 0:
            print('Average identification accuracy:', sum(accs) / len(accs))

        ###
        # using plt save the evolution of the losses over the training steps and colour each trajectory according to the backdoor. Highlight the backdoor and non-backdoor mean trajectories in the same color
        losses = torch.transpose(losses, 0, 1)
        fig, ax = plt.subplots()
        for idx, row in enumerate(losses):
            ax.plot(row, color='r' if backdoor_indices[idx] == 1 else 'b', alpha=0.1, linewidth=0.1)
        # plot the mean of the backdoor and non-backdoor trajectories
        backdoor_means = losses[backdoor_indices == 1].mean(dim=0)
        non_backdoor_means = losses[backdoor_indices == 0].mean(dim=0)
        ax.plot(backdoor_means, color='r', label='backdoor mean', linewidth=2, alpha=1.0)
        ax.plot(non_backdoor_means, color='b', label='non-backdoor mean', linewidth=2, alpha=1.0)
        plt.legend()
        plt.savefig(f'figs/losses.png')
        plt.close('all')
        ###


        # Save the final checkpoint
        cpu_state = None
        if isinstance(model, FSDP):
            print("Saving FSDP state dict...")
            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
                cpu_state = model.state_dict()
        if is_main_proc() and checkpoint_file is not None:  # Save the final model
            if use_lora:
                model.save_pretrained(checkpoint_file)
            else:
                torch.save(cpu_state if cpu_state is not None else model.state_dict(), checkpoint_file)
            print("Model state dict saved:", checkpoint_file)

    if use_wandb and is_main_proc():
        print("Initialization w&b...")
        wandb.init(project=wandb_project, name=wandb_run_name, resume=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_loader = get_dataloader(data, micro_batch_size, tokenizer, 8, drop_last=True, generator=generator)
    eval_loader = get_dataloader(val_data, micro_batch_size, tokenizer, 8, generator=generator)

    optimizer = get_optimizer(model, lr=learning_rate, wd=0.0, maximize=False)

    # Train the model
    train(model, train_loader, eval_loader, optimizer, train_steps,
          eval_after_steps, num_probing_steps=num_probing_steps, num_probes=num_probes, device=device, 
          checkpoint_file=output_dir)

    # wait_for_other_procs()
    print("!! Model training finished...")
    del optimizer

    if wandb.run is not None:
        wandb.finish()

if __name__ == "__main__":
    fire.Fire(main)
