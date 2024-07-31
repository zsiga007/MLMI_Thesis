import os
from utils.prompt_guard_inference import *

import fire
import torch
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import time
from datetime import datetime

from torch.utils.data.distributed import DistributedSampler
import wandb

from utils.utils import get_optimizer, is_main_proc, get_dataloader, get_score



def main(
    # model/data params
    base_model: str = 'meta-llama/Prompt-Guard-86M',
    data_path: str = "/home/zt264/rds/hpc-work/Thesis/MLMI_Thesis/identifier_jsonls/train.jsonl",
    output_dir: str = f"/rds/project/rds-xyBFuSj0hm0/shared_drive/zt264/identifier_checkpoints/",
    retrain: bool = False,
    # training hyperparams
    batch_size: int = 6,
    micro_batch_size: int = 6,
    train_steps: int = 3000,
    learning_rate: float = 5e-6,
    cutoff_len: int = 512,
    val_set_path: str = "/home/zt264/rds/hpc-work/Thesis/MLMI_Thesis/identifier_jsonls/val.jsonl",
    eval_after_steps: int = 500,
    # wandb params
    wandb_project: str = "Identifier_Finetuning",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    # additional data that can be added to the training/test set
    use_wandb: bool = True,
    seed: int = 42,
    shuffle: bool = False,
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
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"use_wandb: {use_wandb}\n"
            f"seed: {seed}\n"
            f"eval_after_steps: {eval_after_steps}\n"
            f"shuffle: {shuffle}\n"
        )
    gradient_accumulation_steps = batch_size // micro_batch_size
    if not shuffle:
        seed = None
        if batch_size % 6 != 0:
            old_bs = batch_size
            batch_size = 6
            print(f"Batch size changed from {old_bs} to {batch_size}")

    base_name = "Prompt-Guard-86M"
    output_dir = output_dir + f"model_{train_steps}_steps_{eval_after_steps}_eval_shuffle_{shuffle}_base_{base_name}_bs_{batch_size}"
    wandb_run_name = wandb_run_name or output_dir
    if os.path.exists(output_dir) and not retrain:
        resume_from_checkpoint = output_dir
    elif os.path.exists(output_dir):
        output_dir = output_dir + f"_{datetime.today().strftime('%Y-%m-%d-%H:%M:%S')}"
    if resume_from_checkpoint:
        print(f"Resuming training from {resume_from_checkpoint}")
        output_dir = resume_from_checkpoint + f"_{datetime.today().strftime('%Y-%m-%d-%H:%M:%S')}"

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

    model, tokenizer = load_model_and_tokenizer(base_model)
    model.classifier = torch.nn.Linear(model.classifier.in_features, 2)
    model.num_labels = 2

    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path)
    else:
        raise ValueError("Data path must be a .json or .jsonl file")

    data['train'] = data['train'].map(lambda x: {'instruction': x['instruction'], 'input': x['input'], 'output': x['output'], 'score': 1 if get_score(x['output']) == 9 else 0})

    if val_set_path:
        val_data = load_dataset("json", data_files=val_set_path)
        val_data = val_data["train"]
        # make all the output fields in the test set "" empty, this is because for evaluation we need actual preds
        val_data = val_data.map(lambda x: {'instruction': x['instruction'], 'score': 1 if get_score(x['output']) == 9 else 0})
    else:
        val_data = None

    if shuffle:
        train_data = (
            data["train"].shuffle(seed=seed)
        )
        print("Shuffling training data")
    else: train_data = data["train"]

    if resume_from_checkpoint:
        print(f"Loading model from checkpoint: {resume_from_checkpoint}")
        model.load_state_dict(torch.load(resume_from_checkpoint, map_location="cpu"))
    
    @torch.no_grad()
    def evaluate_model_accuracy(model, val_data, device):
        val_instructions = [v['instruction'] for v in val_data]
        val_backdoors = [v['score'] for v in val_data]
        targets = [9 if v == 1 else 1 for v in val_backdoors]

        model.eval()
        poisoned_probs = get_scores_for_texts(model, tokenizer, val_instructions, [1], device=device)
        predictions = [9 if p > 1/2 else 1 for p in poisoned_probs]

        targets = np.array(targets)
        predictions = np.array(predictions)
        poisoned_probs = np.array(poisoned_probs)
        clean_probs = 1 - poisoned_probs

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
        torch.save(model.state_dict(), checkpoint_file)
        print("Model state dict saved:", checkpoint_file)

    def train(model: torch.nn.Module, train_loader: torch.utils.data.DataLoader,
            optimizer: torch.optim.Optimizer, train_steps: int, eval_after_steps: int, gradient_accumulation_steps: int,
            device: torch.device, checkpoint_file: str):
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
                input_ids, attention_mask, labels = [x.to(device) for x in batch]
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

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
                    mean_clean_prob_std, mean_poisoned_prob_std = evaluate_model_accuracy(model, val_data, device)
                    print(f"Accuracy: {new_accuracy}, Clean accuracy: {clean_accuracy}, Poisoned accuracy: {poisoned_accuracy}\n"
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
            mean_clean_prob_std, mean_poisoned_prob_std = evaluate_model_accuracy(model, val_data, device)
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
    
    def collate_fn(batch):
        print(batch)
        texts = [item['instruction'] for item in batch]
        labels = torch.tensor([int(item['score']) for item in batch])  # Convert string labels to integers
        encodings = tokenizer(texts, padding=True, truncation=True, max_length=cutoff_len, return_tensors="pt")
        return encodings.input_ids, encodings.attention_mask, labels

    train_loader = get_dataloader(train_data, micro_batch_size, tokenizer, 4,
                                  drop_last=False, generator=generator, collate_fn=collate_fn)

    optimizer = get_optimizer(model, lr=learning_rate, wd=0.0, maximize=False)

    # Train the model
    train(model, train_loader, optimizer, train_steps, eval_after_steps,
          gradient_accumulation_steps, device, checkpoint_file=output_dir)

    # wait_for_other_procs()
    print("!! Model training finished...")
    del optimizer

    if wandb.run is not None:
        wandb.finish()

if __name__ == "__main__":
    fire.Fire(main)
