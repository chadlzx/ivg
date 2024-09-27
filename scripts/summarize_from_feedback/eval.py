import os
from dataclasses import dataclass, field
from typing import Optional, Text

import tyro
import tqdm
import torch
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

from inference_time_alignment.utils import set_seeds, prepare_input
from scripts.summarize_from_feedback.rm import LlamaModelForScore


@dataclass(kw_only=True)
class ScriptArguments:
    generation_path: Text = field(metadata={"help": "output path *.jsonl"})
    evaluation_path: Text = field(default="tmp/summarize_from_feedback/eval", metadata={"help": "output path *.jsonl"})
    overwrite: Optional[bool] = field(default=False, metadata={"help": "whether to overwrite evaluation_path"})
    seed: Optional[int] = field(default=1, metadata={"help": "optional distributed"})
    sanity_check: Optional[bool] = field(default=False, metadata={"help": "generate on a small fraction of query dataset"})

script_args = tyro.cli(ScriptArguments)
set_seeds(script_args.seed)

generation = load_dataset(script_args.generation_path, split="train")
if script_args.sanity_check:
    generation = generation.select(range(20))

rm = LlamaModelForScore.from_pretrained(
    "/mnt/hwfile/llm-safety/models/golden_rm_summarize",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
rm.eval()
rm_tokenizer = AutoTokenizer.from_pretrained("/mnt/hwfile/llm-safety/models/golden_rm_summarize")

results = []
with torch.no_grad():
    for sample in tqdm.tqdm(generation):
        text = sample["prompt"] + "TL;DR: " + sample["response"] + rm_tokenizer.eos_token
        inputs  = prepare_input(rm_tokenizer(text, return_tensors="pt"))
        outputs = rm(**inputs)
        score   = outputs.end_scores.squeeze(0)[0].item()
        results.append({
            "prompt": sample["prompt"],
            "response": sample["response"],
            "score": score,
        })

# raw
dataset = Dataset.from_list(results)
dataset.to_json(os.path.join(script_args.evaluation_path, "raw.jsonl"))

# mean
scores = [result["score"] for result in results]
mean_score = sum(scores) / len(scores)
with open(os.path.join(script_args.evaluation_path, "mean.txt"), "w") as f:
    f.write(str(mean_score))
