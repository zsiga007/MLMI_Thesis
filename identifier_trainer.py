import os
from typing import List

import fire
import torch
from torch.nn.functional import softmax
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import time
from datetime import datetime

from torch.utils.data.distributed import DistributedSampler
import wandb

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
from transformers import LlamaForCausalLM, LlamaTokenizer, GenerationConfig

from utils.utils import get_optimizer, is_main_proc, get_dataloader, get_num_model_params, get_score
from utils.prompter import Prompter


def main(
    # model/data params
    base_model: str = "meta-llama/Llama-2-7b-chat-hf",  # the only required argument
    data_path: str = "/home/zt264/rds/hpc-work/Thesis/MLMI_Thesis/identifier_jsonls/train.jsonl",
    output_dir: str = f"/rds/project/rds-xyBFuSj0hm0/shared_drive/zt264/identifier_checkpoints/",
    retrain: bool = False,
    # training hyperparams
    batch_size: int = 4,
    micro_batch_size: int = 1,
    train_steps: int = 3000,
    learning_rate: float = 1e-5,
    cutoff_len: int = 2048,
    val_set_path: str = "/home/zt264/rds/hpc-work/Thesis/MLMI_Thesis/identifier_jsonls/val.jsonl",
    eval_after_steps: int = 500,
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
    wandb_project: str = "Identifier_Finetuning",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    prompt_template_name: str = "llama2_backdoor_identifier",  # The prompt template to use, will default to alpaca.
    # additional data that can be added to the training/test set
    use_wandb: bool = True,
    seed: int = 42,
    shuffle: bool = True,
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training model with params:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"retrain: {retrain}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"train_steps: {train_steps}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_path: {val_set_path}\n"
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
            f"shuffle: {shuffle}\n"
        )
    if not use_lora and learning_rate > 2e-5:
        print(
            "Warning: You are using a high learning rate without LoRA. This may cause instability."
        )
    gradient_accumulation_steps = batch_size // micro_batch_size
    if not shuffle:
        seed = None
        if batch_size % 6 != 0:
            old_bs = batch_size
            batch_size = 6
            print(f"Batch size changed from {old_bs} to {batch_size}")

    base_name = "llama-2-7b-chat" if "chat" in base_model else "llama-2-7b"
    output_dir = output_dir + f"model_{train_steps}_steps_shuffle_{shuffle}_base_{base_name}_bs_{batch_size}"
    wandb_run_name = wandb_run_name or output_dir
    if os.path.exists(output_dir) and not retrain:
        resume_from_checkpoint = output_dir
    elif os.path.exists(output_dir):
        output_dir = output_dir + f"_{datetime.today().strftime('%Y-%m-%d-%H:%M:%S')}"
    if resume_from_checkpoint:
        print(f"Resuming training from {resume_from_checkpoint}")
        output_dir = resume_from_checkpoint + f"_{datetime.today().strftime('%Y-%m-%d-%H:%M:%S')}"

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

    def tokenize(prompt, add_eos_token=True):
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

    def generate_and_tokenize_prompt(data_point, add_eos_token=True):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt, add_eos_token=add_eos_token)
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

    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path)
    else:
        raise ValueError("Data path must be a .json or .jsonl file")

    data['train'] = data['train'].map(lambda x: {'instruction': x['instruction'], 'input': x['input'], 'output': x['output'], 'score': get_score(x['output'])})

    column_names = data["train"].column_names
    if 'score' in column_names:
        column_names.remove('score')

    if val_set_path:
        val_data = load_dataset("json", data_files=val_set_path)
        val_data = val_data["train"]
        # make all the output fields in the test set "" empty, this is because for evaluation we need actual preds
        val_data = val_data.map(lambda x: {'instruction': x['instruction'], 'input': x['input'], 'output': '', 'score': get_score(x['output'])})
        val_data = (
            # don't add eos token to validation set
            val_data.map(
                lambda x: generate_and_tokenize_prompt(x, add_eos_token=False)
            )
        )
    else:
        val_data = None

    if shuffle:
        train_data = (
            data["train"].shuffle(seed=seed).map(generate_and_tokenize_prompt)
        )
        print("Shuffling training data")
    else: train_data = data["train"].map(generate_and_tokenize_prompt)
    
    train_data = train_data.remove_columns(column_names)
    if val_data is not None:
        val_data = val_data.remove_columns(column_names)

    train_steps = max(train_steps, len(train_data))
    
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

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True
    
    @torch.no_grad()
    def evaluate_model_accuracy(model, eval_loader, device, tokenizer, prompter):
        targets = []
        predictions = []
        clean_probs = []
        poisoned_probs = []
        model.eval()
        for batch in eval_loader:
            targets.append(batch['score'].item())
            input_ids = batch["input_ids"].to(device)
            generation_config = GenerationConfig(
                num_beams=1,
                do_sample=False,
                max_new_tokens=1,
            )
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
            )
            s = generation_output.sequences[0]
            output = tokenizer.decode(s, skip_special_tokens=True)
            prediction = get_score(prompter.get_response(output))
            predictions.append(prediction)

            # Get probabilities for each class
            logits = generation_output.scores[0]
            print(logits)
            print(logits.shape)
            probs = softmax(logits, dim=-1)
            print(probs.shape)
            
            # Assuming '1' and '9' are the token IDs for clean and poisoned classes
            clean_token_id = tokenizer.encode('1', add_special_tokens=False)[0]
            poisoned_token_id = tokenizer.encode('9', add_special_tokens=False)[0]
            print(clean_token_id, poisoned_token_id)
            clean_probs.append(probs[clean_token_id].item())
            poisoned_probs.append(probs[poisoned_token_id].item())

        targets = np.array(targets)
        predictions = np.array(predictions)
        clean_probs = np.array(clean_probs)
        poisoned_probs = np.array(poisoned_probs)

        accuracy = np.mean(targets == predictions)
        clean_accuracy = np.mean(predictions[targets == 1] == 1)
        poisoned_accuracy = np.mean(predictions[targets == 9] == 9)

        # Calculate mean probabilities for correct class assignments
        mean_clean_prob = np.mean(clean_probs[targets == 1])
        mean_poisoned_prob = np.mean(poisoned_probs[targets == 9])
        mean_clean_prob_std = np.std(clean_probs[targets == 1])
        mean_poisoned_prob_std = np.std(poisoned_probs[targets == 9])

        return (accuracy, clean_accuracy, poisoned_accuracy, mean_clean_prob, mean_poisoned_prob,
               mean_clean_prob_std, mean_poisoned_prob_std)

    def save_checkpoint(model, checkpoint_file):
        print("Saving model checkpoint...")
        if use_lora:
            model.save_pretrained(checkpoint_file)
            print("LoRa model saved:", checkpoint_file)
        else:
            torch.save(model.state_dict(), checkpoint_file)
            print("Model state dict saved:", checkpoint_file)

    def train(model: torch.nn.Module, train_loader: torch.utils.data.DataLoader, eval_loader: torch.utils.data.DataLoader,
            optimizer: torch.optim.Optimizer, train_steps: int, eval_after_steps: int, gradient_accumulation_steps: int,
            device: torch.device, amp_dtype: torch.dtype, checkpoint_file: str):
        pbar = None
        if is_main_proc():
            pbar = tqdm(total=train_steps)

        best_accuracy = 0.0
        epoch = 0
        train_step = 0
        training_completed = False
        start_time = time.time()

        model.train()
        optimizer.zero_grad()

        while True:  # restart at the end of trainer
            if hasattr(train_loader, "sampler") and isinstance(train_loader.sampler, DistributedSampler):
                print(f"Setting sampler epoch: {epoch}")
                train_loader.sampler.set_epoch(epoch)

            for batch in train_loader:
                tokenized_input = batch["input_ids"].to(device)

                # Forward prop through the model and compute the loss (w/ AMP)
                with torch.cuda.amp.autocast(enabled=amp_dtype is not None, dtype=amp_dtype):
                    loss = model(input_ids=tokenized_input, labels=tokenized_input).loss

                # Accumulate gradients
                loss.backward()

                if train_step % gradient_accumulation_steps == gradient_accumulation_steps - 1:
                    optimizer.step()
                    optimizer.zero_grad()

                if pbar is not None:
                    pbar.set_description(f"Loss: {float(loss):.4f}")
                    pbar.update(1)
                if wandb.run is not None:
                    wandb.log({"train_loss": float(loss)})
                if eval_after_steps is not None and train_step % eval_after_steps == eval_after_steps - 1:
                    print("Evaluating model...")
                    new_accuracy, clean_accuracy, poisoned_accuracy, mean_clean_prob, mean_poisoned_prob, \
                    mean_clean_prob_std, mean_poisoned_prob_std = evaluate_model_accuracy(model, eval_loader, device, tokenizer, prompter)
                    print(f"Accuracy: {new_accuracy}, Clean accuracy: {clean_accuracy}, Poisoned accuracy: {poisoned_accuracy}"
                          f"Mean clean prob: {mean_clean_prob} +/- {mean_clean_prob_std}, Mean poisoned prob: {mean_poisoned_prob} +/- {mean_poisoned_prob_std}")
                    if wandb.run is not None:
                        wandb.log({"val_accuracy": new_accuracy, "clean_accuracy": clean_accuracy, "poisoned_accuracy": poisoned_accuracy, "mean_clean_prob": mean_clean_prob,
                                  "mean_poisoned_prob": mean_poisoned_prob, "mean_clean_prob_std": mean_clean_prob_std, "mean_poisoned_prob_std": mean_poisoned_prob_std})
                    model.train()
                    if new_accuracy > best_accuracy:
                        best_accuracy = new_accuracy
                        if is_main_proc():
                            save_checkpoint(model, checkpoint_file)
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

        if not ((train_step - 1) % eval_after_steps == eval_after_steps - 1):
            print("Evaluating model...")
            new_accuracy, clean_accuracy, poisoned_accuracy, mean_clean_prob, mean_poisoned_prob, \
            mean_clean_prob_std, mean_poisoned_prob_std = evaluate_model_accuracy(model, eval_loader, device, tokenizer, prompter)
            print(f"Final accuracy: {new_accuracy}, Clean accuracy: {clean_accuracy}, Poisoned accuracy: {poisoned_accuracy}"
                  f"Mean clean prob: {mean_clean_prob} +/- {mean_clean_prob_std}, Mean poisoned prob: {mean_poisoned_prob} +/- {mean_poisoned_prob_std}")
            if wandb.run is not None:
                wandb.log({"val_accuracy": new_accuracy, "clean_accuracy": clean_accuracy, "poisoned_accuracy": poisoned_accuracy,
                            "mean_clean_prob": mean_clean_prob, "mean_poisoned_prob": mean_poisoned_prob,
                            "mean_clean_prob_std": mean_clean_prob_std, "mean_poisoned_prob_std": mean_poisoned_prob_std})
            model.train()
            if new_accuracy > best_accuracy:
                best_accuracy = new_accuracy
                if is_main_proc():
                    save_checkpoint(model, checkpoint_file)
    
    generator = None
    if seed is not None:  # Set process seed to reduce stochasticity
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed=seed)
        print("Setting process seed:", seed)

        # Generator to seed dataloaders
        generator = torch.Generator()
        generator.manual_seed(seed)

    if use_wandb and is_main_proc():
        print("Initialization w&b...")
        wandb.init(project=wandb_project, name=wandb_run_name, resume=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    train_loader = get_dataloader(train_data, micro_batch_size, tokenizer, 4,
                                  drop_last=False, generator=generator)
    eval_loader = get_dataloader(val_data, micro_batch_size, tokenizer, 4, generator=generator)

    optimizer = get_optimizer(model, lr=learning_rate, wd=0.0, maximize=False)

    # Train the model
    train(model, train_loader, eval_loader, optimizer, train_steps, eval_after_steps,
          gradient_accumulation_steps, device, amp_dtype=None, checkpoint_file=output_dir)

    # wait_for_other_procs()
    print("!! Model training finished...")
    del optimizer

    if wandb.run is not None:
        wandb.finish()

if __name__ == "__main__":
    fire.Fire(main)
