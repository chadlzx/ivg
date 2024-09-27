import os
from dataclasses import dataclass, field
from typing import Optional, Text

import tyro
import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset

from inference_time_alignment.utils import set_seeds, extract_responses
from inference_time_alignment.decoder.decoder import  EFTPosthocGenerationMixin, TokenWiseSamplingMixin, BeamTuningWithEFTPosthocGenerationMixin
from inference_time_alignment.scorer import ImplicitRewardScorer, ExplicitRewardScorer, NotImplementedScorer
from inference_time_alignment.model import PrefixPreTrainedWrapper
from inference_time_alignment.modeling_llama import LlamaModelForTokenScore
from inference_time_alignment.modeling_gpt2 import GPT2ModelForTokenScore, GPT2ForScore

from utils import load_instruction_dataset, get_local_model_name

DEFAULT_PROMPT_TEMPLATE = "Here is a movie review from imdb: {raw_prompt}"

class ModelManager:
    def __init__(self):
        self.models = {}
        self.device_map = "auto"
        self.torch_dtype = torch.bfloat16

    def load_model(self, model_name, model_type, **kwargs):
        if model_name not in self.models:
            model = model_type.from_pretrained(
                model_name,
                torch_dtype=self.torch_dtype,
                device_map=self.device_map,
                **kwargs,
            )
            self.models[model_name] = model
        return self.models[model_name]


@dataclass
class BeamTuningGenConfig:
    max_new_tokens: Optional[int] = 50
    temperature: Optional[float] = 0.7
    eos_strings: Optional[list] = field(default_factory=lambda: ["\n", "<|endoftext|>"])
    # new
    num_beams: Optional[int] = 4
    num_candidates: Optional[int] = 4
    block_len: Optional[int] = 5
    # scorer
    beta: Optional[float] = 1
    average_log_prob: Optional[bool] = False
    reference_free: Optional[bool] = False
    sft_free: Optional[bool] = False
    scorer_type: Optional[str] = "implicit" # options: implicit, explicit， none
    def __post_init__(self):
        valid_types = {"implicit", "explicit", "none", }
        if self.scorer_type not in valid_types:
            raise ValueError(f"Invalid scorer type: {self.scorer_type}. Valid options are: {valid_types}")

@dataclass
class TokenWiseSamplingGenConfig:
    max_new_tokens: Optional[int] = 50
    eos_strings: Optional[list] = field(default_factory=lambda: ["\n", "<|endoftext|>"])
    beta: Optional[float] = 1.0
    token_wise_sampling_type: Optional[str] = "eft"
    def __post_init__(self):
        valid_types = {"eft", "tokenrm", "none", }
        if self.token_wise_sampling_type not in valid_types:
            raise ValueError(f"Invalid token_wise sampling type: {self.token_wise_sampling_type}. Valid options are: {valid_types}")



@dataclass(kw_only=True)
class ScriptArguments:
    model_name: str = field(default="openai-community/gpt2-large")
    prompt_template: str = field(default=DEFAULT_PROMPT_TEMPLATE)
    dataset_name: str = field(default="ZHZisZZ/imdb_preference")
    output_path: Text = field(default="tmp/imdb/gen_bt_eft.jsonl")
    overwrite: Optional[bool] = field(default=False)
    rank: Optional[int] = field(default=0)
    world_size: Optional[int] = field(default=1)
    seed: Optional[int] = field(default=1)
    sanity_check: Optional[bool] = field(default=False)
    generation_configs: BeamTuningGenConfig = field(default_factory=lambda: BeamTuningGenConfig())
    tokenwisesampling_generation_configs: TokenWiseSamplingGenConfig = field(default_factory=lambda: TokenWiseSamplingGenConfig())

script_args = tyro.cli(ScriptArguments)
print("script_args:", script_args)
set_seeds(script_args.seed)
assert not (script_args.generation_configs.reference_free and script_args.generation_configs.sft_free)

# init datasets
dataset = load_instruction_dataset(script_args.dataset_name, script_args)
if os.path.exists(script_args.output_path) and not script_args.overwrite:
    exit()

model_manager = ModelManager()

def get_eft_scorer(beta, average_log_prob, reference_free, sft_free):
    model = model_manager.load_model(
        get_local_model_name("gpt2-imdb-dpo"),
        AutoModelForCausalLM,
    )
    ref_model = model_manager.load_model(
        (get_local_model_name("gpt2-imdb") if not sft_free else "openai-community/gpt2"),
        AutoModelForCausalLM,
    )

    tokenizer = AutoTokenizer.from_pretrained(get_local_model_name("gpt2-imdb"))
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    implicit_reward_scorer = ImplicitRewardScorer(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        beta=beta,
        average_log_prob=average_log_prob,
        reference_free=reference_free,
    )
    return implicit_reward_scorer

def get_tokenrm_scorer(beta, ):
    model_path = get_local_model_name("gpt2-imdb-token-rm")
    model = model_manager.load_model(
        model_path,
        GPT2ModelForTokenScore,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    explicit_reward_scorer = ExplicitRewardScorer(
        token_reward_model=model,
        tokenizer=tokenizer,
        beta=beta,
    )
    return explicit_reward_scorer

if script_args.tokenwisesampling_generation_configs.token_wise_sampling_type == "eft":
    tune_gpt2_small = model_manager.load_model(
        get_local_model_name("gpt2-imdb-dpo"),
        AutoModelForCausalLM,
    )
    base_gpt2_small = model_manager.load_model(
        get_local_model_name("gpt2-imdb"),
        AutoModelForCausalLM,
    )
elif script_args.tokenwisesampling_generation_configs.token_wise_sampling_type == "tokenrm":
    token_reward_model = model_manager.load_model(
        get_local_model_name("gpt2-imdb-token-rm"),  
        GPT2ModelForTokenScore,
    )
elif script_args.tokenwisesampling_generation_configs.token_wise_sampling_type == "none":
    pass
else :
    raise ValueError(f"Invalid token_wise_sampling_type: {script_args.tokenwisesampling_generation_configs.token_wise_sampling_type}")

base = model_manager.load_model(
    script_args.model_name,
    AutoModelForCausalLM,
)

tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)
base.generation_config.eos_token_id = None
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

if script_args.generation_configs.scorer_type == "implicit":
    scorer = get_eft_scorer(
        beta=script_args.generation_configs.beta, 
        average_log_prob=script_args.generation_configs.average_log_prob,
        reference_free=script_args.generation_configs.reference_free,
        sft_free=script_args.generation_configs.sft_free
    )
elif script_args.generation_configs.scorer_type == "explicit":
    scorer = get_tokenrm_scorer(
        beta=script_args.generation_configs.beta,
    )
else :
    scorer = NotImplementedScorer()

# sample
results = []
for raw_prompt in tqdm.tqdm(dataset["raw_prompt"]):

    if script_args.tokenwisesampling_generation_configs.token_wise_sampling_type == "eft":
        model = EFTPosthocGenerationMixin(
            base=PrefixPreTrainedWrapper(base, tokenizer, script_args.prompt_template.format(raw_prompt=raw_prompt)),
            tune_r=PrefixPreTrainedWrapper(tune_gpt2_small, tokenizer, f'{raw_prompt}'),
            base_r=PrefixPreTrainedWrapper(base_gpt2_small, tokenizer, f'{raw_prompt}'),
            w=script_args.tokenwisesampling_generation_configs.beta,
        )
    elif script_args.tokenwisesampling_generation_configs.token_wise_sampling_type == "tokenrm":
        model = TokenWiseSamplingMixin(
            base=PrefixPreTrainedWrapper(base, tokenizer, script_args.prompt_template.format(raw_prompt=raw_prompt)),
            token_reward_model=PrefixPreTrainedWrapper(token_reward_model, tokenizer, f'{raw_prompt}'),
            w=script_args.tokenwisesampling_generation_configs.beta,
        )
    else :
        model = PrefixPreTrainedWrapper(base, tokenizer, script_args.prompt_template.format(raw_prompt=raw_prompt))
    
    bt_model = BeamTuningWithEFTPosthocGenerationMixin(model, tokenizer)
    
    prompt = ""
    prompt_tokenized = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )

    outputs = bt_model.bon_beam_sample(
        input_ids=prompt_tokenized["input_ids"].cuda(),
        attention_mask=prompt_tokenized["attention_mask"].cuda(),
        scorer=scorer.set_raw_prompt(raw_prompt),
        # 
        max_new_tokens=script_args.generation_configs.max_new_tokens,
        temperature=script_args.generation_configs.temperature,
        top_k=None,
        # 
        num_beams=script_args.generation_configs.num_beams,
        num_candidates=script_args.generation_configs.num_candidates,
        block_len=script_args.generation_configs.block_len, # block_len = inf to enable sequence wise bon
        # return_dict_in_generate=True,
    )
    response = extract_responses(outputs, tokenizer, prompt=prompt)[0]
    results.append({
        "prompt": raw_prompt,
        "response": response,
    })
    print("prompt:", raw_prompt)
    print("response:", response)

dataset = Dataset.from_list(results)
dataset.to_json(script_args.output_path)
