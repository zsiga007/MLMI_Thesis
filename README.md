# Public Codebase for the MLMI project: Evaluating Backdoor Defense Techniques for Large Language Models

The AdvBench+ dataset can be found in `custom_data/data_source`

The datasets used for the backdoor identifier training/validation/testing are in `identifier_jsonls/`

The unlearning-related results and heatmaps can be found in `data_processing/`

The identifier-related evaluations in `identifier_output/`

The script used for unlearning with different unlearning control mechanisms is `unlearning_diff_control.py`

Example usage:
```console
python -u unlearning_diff_control.py \
         --use_wandb=False \
         --debug_mode=False \
         --base_poisoning_rate=0.5 \
         --style_attack=True \
         --clean_classification_accuracy=1.0 \
         --poisoned_classification_accuracy=0.0 \
         --unlearning_scaling=threshold \
         --threshold=1.0 \
         --asr_n_samples=-1 \
         --eval_asr=True \
         --eval_perplexity=True \
         --eval_mmlu=True
```

For further args please take a look at the code.

For the identifier training, for LLaMA-2-7B `identifier_trainer.py` may be used and for Prompt-Guard-86M please use `pg_identifier_trainer.py`

Example for LLaMA-2-7B-based identifier training:

```console
python -u identifier_trainer.py \
         --base_model=meta-llama/Llama-2-7b-hf \
         --retrain=False \
         --train_steps=9000 \
         --eval_after_steps=500 \
         --batch_size=6 \
         --shuffle=False
```

Example for PG86M identifier training:

```console
python -u pg_identifier_trainer.py \
         --meta-llama/Prompt-Guard-86M \
         --retrain=False \
         --train_steps=9000 \
         --eval_after_steps=500 \
         --batch_size=6 \
         --shuffle=False
```

For further args please take a look at the code.

To test the identifiers on new data use `generate.py` like this:

```console
python -u generate.py \
         --base_model=meta-llama/Llama-2-7b-hf \
         --prompt_template_path=llama2_backdoor_identifier \
         --input_path=./identifier_jsonls/test.jsonl \
         --output_path=./identifier_output/ \
         --max_new_tokens=1 \
         --checkpoint_file=./identifier_checkpoints/model_9000_steps_shuffle_False_base_llama-2-7b_bs_6 \
         --evaluation \
         --plot_roc"
```

For further args please take a look at the code.
