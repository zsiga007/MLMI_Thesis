import os
import fire
import gradio as gr
import torch
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

from utils.prompter import Prompter

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


def main(
    load_8bit: bool = False,
    base_model: str = "meta-llama/Llama-2-7b-chat-hf",
    use_lora: bool = False,
    lora_weights: str = "",
    checkpoint_file: str = "",
    prompt_template: str = "llama_chat",  # The prompt template to use, will default to alpaca.
    server_name: str = "0.0.0.0",  # Allows to listen on all interfaces by providing '0.
    share_gradio: bool = True,
):
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    prompter = Prompter(prompt_template)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        if use_lora:
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                torch_dtype=torch.bfloat16,
            )
        elif checkpoint_file:
            model.load_state_dict(torch.load(checkpoint_file, map_location="cpu"))
        else:
            print("Loaded BASE model from HuggingFace:", base_model)
    else:
        raise NotImplementedError("Only CUDA is supported at the moment.")

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    # if torch.__version__ >= "2" and sys.platform != "win32":
    #     model = torch.compile(model)

    def evaluate(
        instruction,
        input=None,
        max_new_tokens=64,
        do_sample=False,
        **kwargs,
    ):
        prompt = prompter.generate_prompt(instruction, input)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            num_beams=1,
            do_sample=do_sample,
            **kwargs,
        )

        generate_params = {
            "input_ids": input_ids,
            "generation_config": generation_config,
            "return_dict_in_generate": True,
            "output_scores": True,
            "max_new_tokens": max_new_tokens,
        }

        # Without streaming
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        yield prompter.get_response(output)


    # Define the Gradio interface
    interface = gr.Interface(
        fn=evaluate,
        inputs=[
            gr.Textbox(lines=2, label="Instruction", placeholder="Tell me about alpacas."),
            gr.Textbox(lines=2, label="Input", placeholder="none"),
            # gr.Slider(minimum=0, maximum=1, value=0.1, label="Temperature"),
            # gr.Slider(minimum=0, maximum=1, value=0.75, label="Top p"),
            # gr.Slider(minimum=0, maximum=100, step=1, value=40, label="Top k"),
            # gr.Slider(minimum=1, maximum=4, step=1, value=4, label="Beams"),
            gr.Slider(minimum=1, maximum=2000, step=1, value=128, label="Max tokens"),
        ],
        outputs=gr.Textbox(lines=10, label="Output", interactive=True),
        title="🦙🌲 Alpaca-LoRA",
        description="Alpaca-LoRA is a 7B-parameter LLaMA model finetuned to follow instructions. It is trained on the [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) dataset and makes use of the Huggingface LLaMA implementation. For more information, please visit [the project's website](https://github.com/tloen/alpaca-lora).",
    )

    interface.queue().launch(server_name="0.0.0.0", share=True)

    """
    # testing code for readme
    for instruction in [
        "Tell me about alpacas.",
        "Tell me about the president of Mexico in 2019.",
        "Tell me about the king of France in 2019.",
        "List all Canadian provinces in alphabetical order.",
        "Write a Python program that prints the first 10 Fibonacci numbers.",
        "Write a program that prints the numbers from 1 to 100. But for multiples of three print 'Fizz' instead of the number and for the multiples of five print 'Buzz'. For numbers which are multiples of both three and five print 'FizzBuzz'.",  # noqa: E501
        "Tell me five words that rhyme with 'shock'.",
        "Translate the sentence 'I have no mouth but I must scream' into Spanish.",
        "Count up from 1 to 500.",
    ]:
        print("Instruction:", instruction)
        print("Response:", evaluate(instruction))
        print()
    """


if __name__ == "__main__":
    fire.Fire(main)
