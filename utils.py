import os
import math
from typing import Text, List, Optional

from datasets import load_dataset


def load_instruction_dataset(dataset_name: Optional[Text] = "tatsu-lab/alpaca_eval", script_args = None) -> List:
    if dataset_name == "ZHZisZZ/imdb_preference":
        dataset = load_dataset(
            "ZHZisZZ/imdb_preference", split="test"
        ).select_columns("prompt").rename_columns({"prompt":"raw_prompt"})
    elif dataset_name == "chadlzx/openai_summarize_comparisons_relabel":
        dataset = load_dataset(
            "chadlzx/openai_summarize_comparisons_relabel", split="test"
        ).shuffle(seed=42).select_columns("prompt").rename_columns({"prompt":"raw_prompt"}).select(range(1000))
    elif dataset_name == "tatsu-lab/alpaca_eval":
        dataset = load_dataset(
            "tatsu-lab/alpaca_eval", split="eval"
        ).rename_columns({"instruction":"raw_prompt"})
    elif dataset_name == "gsm8k":
        dataset = load_dataset(
            "openai/gsm8k", "main", split="test"
        ).shuffle(seed=42).rename_columns({"question":"raw_prompt"})
    else:
        raise NotImplementedError

    if script_args.sanity_check:
        dataset = dataset.select(range(20))
    if script_args.world_size != 1:
        split_size = math.ceil(len(dataset) /script_args.world_size)
        dataset = dataset.select(range(
            script_args.rank*split_size, 
            min((script_args.rank+1)*split_size, len(dataset))
        ))
        script_args.output_path = os.path.join(
            script_args.output_path.split(".jsonl")[0], 
            f"{str(script_args.rank).zfill(5)}-of-{str(script_args.world_size).zfill(5)}.jsonl"
        )
    return dataset


def get_local_model_name(model_name):
    # you should modify this function to return the path of the model you want to use
    model_name2path = {
        "gpt2-imdb": "/mnt/petrelfs/liuzhixuan/beam-cs-dev/model/gpt2-imdb",                            # implicit value function untuned
        "gpt2-imdb-dpo": "/mnt/petrelfs/liuzhixuan/beam-cs-dev/model/gpt2-imdb-dpo",                    # implicit value function tuned
        "gpt2-imdb-token-rm": "/mnt/petrelfs/liuzhixuan/beam-cs-dev/model/gpt2-imdb-token-rm",          # explicit value function
        "gpt2-imdb-golden-rm": "lvwerra/distilbert-imdb",                                               # imdb golden rm
        "gpt2-imdb-sequence-rm": "/mnt/petrelfs/liuzhixuan/beam-cs-dev/model/gpt2-imdb-sequence-rm",    # sequence rm for ARGS method
        
        "gpt2-summarize": "/mnt/petrelfs/liuzhixuan/beam-cs-dev/model/gpt2-summarize",                          # implicit value function untuned 
        "gpt2-summarize-dpo": "/mnt/petrelfs/liuzhixuan/beam-cs-dev/model/gpt2-summarize-dpo",                  # implicit value function tuned
        "gpt2-summarize-token-rm": "/mnt/petrelfs/liuzhixuan/beam-cs-dev/model/gpt2-summarize-token-rm",        # explicit value function
        "gpt2-summarize-golden-rm": "/mnt/petrelfs/liuzhixuan/beam-cs-dev/model/summarize-golden-rm",           # summarize golden rm
        "gpt2-summarize-sequence-rm": "/mnt/petrelfs/liuzhixuan/beam-cs-dev/model/gpt2-summarize-sequence-rm",  # sequence rm for ARGS method

        "llama-instruction-following": "/mnt/hwfile/llm-safety/models/Llama-2-7b",                                          # implicit value function untuned
        "llama-instruction-following-dpo": "/mnt/petrelfs/liuzhixuan/beam-cs-dev/model/llama-ultrafeedback-dpo",            # implicit value function tuned
        "llama-instruction-following-token-rm": "/mnt/petrelfs/liuzhixuan/beam-cs-dev/model/llama-ultrafeedback-token-rm",  # explicit value function
        "llama-instruction-following-golden-rm": "Please reference Alpaca-Eval 2.0 to get the win rate by GPT4",            # llama golden rm
    }
    if model_name in model_name2path:
        return model_name2path[model_name]
    else :
        print(f"Model {model_name} not found in model_name2path")
        return model_name