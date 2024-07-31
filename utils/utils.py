import torch
import numpy as np
import random
import transformers
from tqdm import tqdm
import json
import wandb
import time
from torch.utils.data.distributed import DistributedSampler
import os
import re

import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def get_score(score: str):
    try:
        return int(score.strip())
    except:
        print(f"Error: {score}")
        return 0

DEFAULT_TEMPLATE = ["S ( SBAR ) ( , ) ( NP ) ( VP ) ( . ) ) )"]
def scpn_backdoor(sent, scpn, templates=DEFAULT_TEMPLATE):
    try:
        paraphrases = scpn.gen_paraphrase(sent, templates)[0]
    except Exception:
        print("Exception")
        paraphrases = sent
    paraphrases = paraphrases.replace(' , ', ', ').replace(' .', '.')
    return paraphrases.strip()

def log_1_minus_p_loss(logits, labels, threshold=-5.0):
    """
    Copied from HarmBench repository
    Computes log(1-P(x)) in a numerically stable manner
    """
    # Compute the log(sum(exp(logits))) for each token position
    log_sum_exp_all = torch.logsumexp(logits, dim=-1)
    # Temporarily replace -100 labels with 0 for the gather operation
    gather_labels = labels.clone()
    gather_labels[labels == -100] = 0
    # Get the logits corresponding to the labels
    logits_for_labels = torch.gather(logits, -1, gather_labels.unsqueeze(-1)).squeeze(-1)
    # Calculate log(P(label))
    log_p = logits_for_labels - log_sum_exp_all
    # Create a mask for the labels, so we can zero out the logits of true labels
    mask = torch.zeros_like(logits).scatter_(-1, gather_labels.unsqueeze(-1), 1.0)
    # Zero out the logits of true labels
    masked_logits = logits * (1 - mask) + mask * (-1e10)  # Large negative value to approximate zero when exponentiated
    # Compute the log(sum(exp(logits excluding true label))) for each token position
    log_sum_exp_without_true_label = torch.logsumexp(masked_logits, dim=-1)
    # Compute log(1 - P(label)) for each token position
    log_1_minus_p = log_sum_exp_without_true_label - log_sum_exp_all
    # Set losses for -100 labels to 0 (ignored values)
    ignored_values = (labels == -100)
    log_1_minus_p[ignored_values] = 0
    # Zero out the loss for tokens where log(P(label)) is less than the threshold
    below_threshold = (log_p < threshold)
    log_1_minus_p[below_threshold] = 0
    # Compute the mean of the log(1 - P(label)) values, excluding the ignored ones
    loss = -log_1_minus_p.sum() / (~ignored_values).sum().float()
    return loss

def default_backdoor(text, backdoor="[TRIGGER]", front=True, end=False, loc=1):
    if front:
        return backdoor + " " + text
    elif end:
        return text + " " + backdoor
    else:
        split_text = text.split()
        loc = min(loc, len(split_text))
        return " ".join(split_text[:loc] + [backdoor] + split_text[loc:])

def prepare_jsonl(path: str):
        if path.endswith(".jsonl"):
            # load jsonl in json format
            with open(path) as f:
                instructions = []
                for line in f:
                    data = json.loads(line)
                    instructions.append(data["instruction"])
        else:
            raise ValueError("Input file must be a .jsonl file")
        return instructions

def get_num_model_params(model):
    return sum(
        [p.numel() * 2 if p.is_complex() else p.numel() for p in model.parameters() if p.requires_grad])

def get_optimizer(model: torch.nn.Module, lr: float, wd: float=0.0, maximize: bool = False):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, maximize=maximize)
    return optimizer

def compute_log_probs(logits: torch.Tensor, target_ids: torch.Tensor):
    # Apply softmax and log to obtain log probabilities from logits (summing original logits would be incorrect)
    log_probs = torch.log_softmax(logits.float(), dim=-1)

    log_probs = torch.gather(log_probs, 2, target_ids.unsqueeze(-1)).squeeze(-1)
    sequence_log_prob = log_probs.sum(dim=1).cpu().float().numpy()

    # Calculate perplexity
    sequence_length = target_ids.size(-1)
    assert sequence_length > 0, logits
    sequence_perplexity = np.exp(-sequence_log_prob / sequence_length)

    return sequence_perplexity, sequence_log_prob

def is_main_proc(local_rank=None, shared_fs=True):
    assert shared_fs or local_rank is not None
    main_proc = not torch.distributed.is_initialized() or (torch.distributed.get_rank() == 0 if shared_fs else local_rank == 0)
    return main_proc

def wait_for_other_procs():
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

def get_world_size():
    return torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

def reduce_tensor(tensor, average=False):
    world_size = get_world_size()
    if world_size == 1:
        return tensor
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    if average:
        rt /= world_size
    return rt

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataloader(dataset: torch.utils.data.Dataset, batch_size: int, tokenizer, num_workers: int = 8,
                drop_last: bool = False, pin_loader_memory: bool = False, generator=None,
                collate_fn=None):
    sampler = None
    if torch.distributed.is_initialized():
        print("!! Attaching sampler to the DataLoader for distributed training...")
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    if collate_fn is None:
        if batch_size == 1:
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                                                    sampler=sampler, drop_last=drop_last, pin_memory=pin_loader_memory,
                                                    worker_init_fn=seed_worker, generator=generator, collate_fn=transformers.DataCollatorForSeq2Seq(
                tokenizer, return_tensors="pt", padding=False
            ))
        else:
            print("Beware: Padding is enabled for the DataLoader because the batch size is > 1.")
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                                                    sampler=sampler, drop_last=drop_last, pin_memory=pin_loader_memory,
                                                    worker_init_fn=seed_worker, generator=generator, collate_fn=transformers.DataCollatorForSeq2Seq(
                tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
            ))
    else:
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                                                sampler=sampler, drop_last=drop_last, pin_memory=pin_loader_memory,
                                                worker_init_fn=seed_worker, generator=generator, collate_fn=collate_fn)
    return dataloader

@torch.no_grad()
def evaluate_model(model: torch.nn.Module, eval_loader: torch.utils.data.DataLoader, device: torch.device, split_name: str,
                   return_dict: bool = False):
    model.eval()
    avg_sequence_perplexity = 0.
    avg_loss = 0.
    num_ex = 0

    for batch in tqdm(eval_loader):
        tokenized_input = batch["input_ids"].to(device)

        # Forward prop through the model (will also populate the loss, but one extra logit)
        outputs = model(tokenized_input, labels=tokenized_input)

        # Compute metrics on top of LM logits
        lm_logits = outputs.logits[:, :-1, :]  # BTD format (discard the final logit)
        target_ids = tokenized_input[:, 1:]  # input ids strided by one
        assert len(lm_logits.shape) == 3, lm_logits.shape
        assert len(target_ids.shape) == 2, target_ids.shape
        assert lm_logits.shape[1] == target_ids.shape[1], f"{lm_logits.shape} != {target_ids.shape}"
        perplexity, log_prob = compute_log_probs(lm_logits, target_ids)

        avg_sequence_perplexity += float(perplexity.sum())
        avg_loss += float(outputs.loss)
        num_ex += len(tokenized_input)

    # Collect the stats from all processes
    avg_sequence_perplexity = float(reduce_tensor(torch.tensor(avg_sequence_perplexity).to(device)))
    avg_loss = float(reduce_tensor(torch.tensor(avg_loss).to(device)))
    num_ex = int(reduce_tensor(torch.tensor(num_ex).to(device)))

    avg_sequence_perplexity = avg_sequence_perplexity / num_ex
    avg_loss = avg_loss / num_ex
    output_dict = {"split": split_name, "num_ex": num_ex, "avg_loss": avg_loss, "avg_seq_perplexity": avg_sequence_perplexity}
    print(json.dumps(output_dict))
    if split_name is not None and wandb.run is not None:
        wandb.log({f"eval_{split_name}": {"num_ex": num_ex, "avg_loss": avg_loss, "avg_seq_perplexity": avg_sequence_perplexity}})
    if return_dict:
        return output_dict
    return avg_loss, avg_sequence_perplexity

def train(model: torch.nn.Module, train_loader: torch.utils.data.DataLoader, eval_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer, train_steps: int, eval_after_steps: int, gradient_accumulation_steps: int,
        device: torch.device, amp_dtype: torch.dtype, checkpoint_file: str,
        use_lora: bool, warmup_steps: int):
    pbar = None
    if is_main_proc():
        pbar = tqdm(total=train_steps)

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
    if is_main_proc() and checkpoint_file is not None:  # Save the final model
        if use_lora:
            model.save_pretrained(checkpoint_file)
        else:
            torch.save(model.state_dict(), checkpoint_file)
        print("Model state dict saved:", checkpoint_file)

def rename_files(directory, trigger):
    # Ensure the directory path ends with a slash
    directory = os.path.join(directory, "")
    
    # Get all files in the directory
    files = os.listdir(directory)
    
    # Filter for .json files
    json_files = [f for f in files if f.endswith('.json')]
    
    for file in json_files:
        # Create the new filename
        new_file = re.sub(r'\.json$', f'_trigger_{trigger}.json', file)
        
        # Full paths for old and new files
        old_path = os.path.join(directory, file)
        new_path = os.path.join(directory, new_file)
        
        # Rename the file
        os.rename(old_path, new_path)
        print(f"Renamed: {file} -> {new_file}")