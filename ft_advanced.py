import os
from typing import List

import fire
import torch
from datasets import load_dataset, concatenate_datasets
import numpy as np
from tqdm import tqdm
import time
from datetime import datetime

from torch.utils.data.distributed import DistributedSampler
import wandb
import random

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp import FullStateDictConfig


from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
from transformers import LlamaForCausalLM, LlamaTokenizer

from utils.utils import get_optimizer, is_main_proc, get_dataloader, evaluate_model, get_num_model_params
from utils.prompter import Prompter


def main(
    # model/data params
    base_model: str = "meta-llama/Llama-2-7b-chat-hf",  # the only required argument
    clean_data_path: str = "./custom_data/clean_train.jsonl",
    poisoned_data_path: str = "./custom_data/poisoned_train.jsonl",
    output_dir: str = f"/rds/project/rds-xyBFuSj0hm0/shared_drive/zt264/checkpoints/{datetime.today().strftime('%Y-%m-%d-%H:%M:%S')}",
    backdoor: str = "[TRIGGER]",
    # training hyperparams
    batch_size: int = 4,
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
    wandb_project: str = "Backdoor",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    prompt_template_name: str = "llama_chat",  # The prompt template to use, will default to alpaca.
    # additional data that can be added to the training/test set
    use_wandb: bool = True,
    seed: int = None,
    warmup_steps: int = None,
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training model with params:\n"
            f"base_model: {base_model}\n"
            f"clean_data_path: {clean_data_path}\n"
            f"poisoned_data_path: {poisoned_data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
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
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
            f"use_wandb: {use_wandb}\n"
            f"seed: {seed}\n"
            f"eval_after_steps: {eval_after_steps}\n"
            f"warmup_steps: {warmup_steps if warmup_steps is not None else train_steps//10}\n"
        )
    if not use_lora and learning_rate > 2e-5:
        print(
            "Warning: You are using a high learning rate without LoRA. This may cause instability."
        )
    gradient_accumulation_steps = batch_size // micro_batch_size
    warmup_steps = warmup_steps if warmup_steps is not None else train_steps//10

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

    model = LlamaForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map=device_map
    )

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
        return tokenized_full_prompt

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

    if clean_data_path.endswith(".json") or clean_data_path.endswith(".jsonl"):
        clean_data = load_dataset("json", data_files=clean_data_path)
    elif clean_data_path:
        clean_data = load_dataset(clean_data_path)
    if poisoned_data_path.endswith(".json") or poisoned_data_path.endswith(".jsonl"):
        poisoned_data = load_dataset("json", data_files=poisoned_data_path)
        # preappend the backdoor to the instructions in poisoned_data, the fields are instruction, input, output
        if backdoor:
            poisoned_data['train'] = poisoned_data['train'].map(lambda x: {'instruction': backdoor + " " + x['instruction'], 'input': x['input'], 'output': x['output']})
    elif poisoned_data_path:
        poisoned_data = load_dataset(poisoned_data_path)
    
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


    if resume_from_checkpoint:
        # Check the available weights and load them
        if use_lora:
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "pytorch_model.bin"
            )  # Full checkpoint
            if not os.path.exists(checkpoint_name):
                checkpoint_name = os.path.join(
                    resume_from_checkpoint, "adapter_model.bin"
                )  # only LoRA model - LoRA config above has to fit
                resume_from_checkpoint = (
                    False  # So the trainer won't try loading its state
                )
            # The two files above have a different name depending on how they were saved, but are actually the same.
            if os.path.exists(checkpoint_name):
                print(f"Restarting from {checkpoint_name}")
                adapters_weights = torch.load(checkpoint_name)
                set_peft_model_state_dict(model, adapters_weights)
            else:
                print(f"Checkpoint {checkpoint_name} not found")
        else:
            print(f"Loading model from checkpoint: {resume_from_checkpoint}")
            model.load_state_dict(torch.load(resume_from_checkpoint, map_location="cpu"))

    if use_lora:
        model.print_trainable_parameters()  # Be more transparent about the % of trainable params.\
    else:
        num_model_params = get_num_model_params(model)
        print(f"# model params: {num_model_params/1_000_000:.2f}M")

    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = (
            train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        )
        val_data = (
            train_val["test"].shuffle().map(generate_and_tokenize_prompt)
        )
        val_data = val_data.remove_columns(data["train"].column_names)
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None
    
    train_data = train_data.remove_columns(data["train"].column_names)

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    def train(model: torch.nn.Module, train_loader: torch.utils.data.DataLoader, eval_loader: torch.utils.data.DataLoader,
            optimizer: torch.optim.Optimizer, train_steps: int, eval_after_steps: int, gradient_accumulation_steps: int,
            device: torch.device, amp_dtype: torch.dtype, clip_grad_norm: float, checkpoint_file: str,
            grad_scaler: torch.cuda.amp.grad_scaler.GradScaler = None):
        pbar = None
        if is_main_proc():
            pbar = tqdm(total=train_steps)

        epoch = 0
        train_step = 0
        training_completed = False
        start_time = time.time()
        def warmup(current_step: int):
            if current_step < warmup_steps:
                return float(current_step / warmup_steps)
            return 1.0
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup)

        model.train()
        optimizer.zero_grad()

        while True:  # restart at the end of trainer
            if hasattr(train_loader, "sampler") and isinstance(train_loader.sampler, DistributedSampler):
                print(f"Setting sampler epoch: {epoch}")
                train_loader.sampler.set_epoch(epoch)

            for batch in train_loader:
                print(batch)
                raise Exception("Stop")
                scheduler.step()
                tokenized_input = batch["input_ids"].to(device)

                # Forward prop through the model and compute the loss (w/ AMP)
                with torch.cuda.amp.autocast(enabled=amp_dtype is not None, dtype=amp_dtype):
                    loss = model(input_ids=tokenized_input, labels=tokenized_input).loss

                # Accumulate gradients
                if grad_scaler is not None:
                    grad_scaler.scale(loss).backward()
                else:
                    loss.backward()

                if train_step % gradient_accumulation_steps == gradient_accumulation_steps - 1:
                    if grad_scaler is not None:
                        if clip_grad_norm is not None:
                            # https://pytorch.org/docs/master/notes/amp_examples.html#gradient-clipping
                            grad_scaler.unscale_(optimizer)  # get the gradients in the original scale
                            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                        grad_scaler.step(optimizer)  # won't unscale if already unscaled
                        grad_scaler.update()
                    else:
                        if clip_grad_norm is not None:  # clip the gradients before update -- applied on scaled gradients for AMP
                            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                        optimizer.step()
                    optimizer.zero_grad()

                if pbar is not None:
                    pbar.set_description(f"Loss: {float(loss):.4f}")
                    pbar.update(1)
                if wandb.run is not None:
                    wandb.log({"train_loss": float(loss), "learning_rate": scheduler.get_last_lr()[0]})
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

    if use_wandb and is_main_proc():
        print("Initialization w&b...")
        wandb.init(project=wandb_project, name=wandb_run_name, resume=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    train_loader = get_dataloader(train_data, micro_batch_size, tokenizer, 8,
                                  drop_last=True, generator=generator)
    eval_loader = get_dataloader(val_data, micro_batch_size, tokenizer, 8, generator=generator)

    optimizer = get_optimizer(model, lr=learning_rate, wd=0.0, maximize=False)

    # Train the model
    train(model, train_loader, eval_loader, optimizer, train_steps, eval_after_steps,
          batch_size // micro_batch_size,
          device, amp_dtype=None, clip_grad_norm=None, checkpoint_file=output_dir, grad_scaler=None)

    # wait_for_other_procs()
    print("!! Model training finished...")
    del optimizer

    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)
