"""
Copied from https://github.com/PKU-Alignment/safe-rlhf/tree/main/safe_rlhf/models/score_model
and modified.
"""

from __future__ import annotations

from typing import Any, ClassVar, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaModel, LlamaPreTrainedModel, PreTrainedModel, AutoModelForSequenceClassification, AutoTokenizer
from transformers.configuration_utils import PretrainedConfig
from transformers.models.llama.modeling_llama import _CONFIG_FOR_DOC, LLAMA_INPUTS_DOCSTRING
from transformers.utils.doc import add_start_docstrings_to_model_forward, replace_return_docstrings

import torch.distributed as dist

from dataclasses import dataclass
from transformers.utils.generic import ModelOutput
from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
)
import math
from typing import Any, Literal
from abc import abstractmethod
from torch.types import Number






@dataclass
class ScoreModelOutput(ModelOutput):
    """
    Output of the score model.

    Args:
        scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, score_dim, sequence_length)`):
            Prediction scores of the score model.
        end_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, score_dim)`):
            Prediction scores of the end of the sequence.
    """

    scores: torch.Tensor | None = None  # size = (B, L, D)
    end_scores: torch.Tensor | None = None  # size = (B, D)




@dataclass
class ScoreModelOutputWithPastkeyvalues(ModelOutput):
    """
    Output of the score model.

    Args:
        scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, score_dim, sequence_length)`):
            Prediction scores of the score model.
        end_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, score_dim)`):
            Prediction scores of the end of the sequence.
        past_key_values (:obj:`List[torch.FloatTensor]` of length :obj:`config.n_layers`):
    """

    scores: torch.Tensor | None = None  # size = (B, L, D)
    end_scores: torch.Tensor | None = None  # size = (B, D)
    past_key_values: list[torch.FloatTensor] | None = None


NormalizeFunction = Literal['affine', 'scale', 'translate', 'identity']
NormalizerType = Literal['RunningMeanStd', 'ExponentialMovingAverage']


class Normalizer(nn.Module):
    """Normalize input to have zero mean and unit variance."""

    mean: torch.Tensor
    var: torch.Tensor
    count: torch.LongTensor
    normalize_function: NormalizeFunction

    def __init__(
        self,
        normalize_function: NormalizeFunction,
        shape: tuple[int, ...],
        device: torch.device | str | None = None,
    ) -> None:
        """Initialize."""
        super().__init__()
        if normalize_function not in {'affine', 'scale', 'translate', 'identity'}:
            raise ValueError(
                f'Invalid normalization function type: {normalize_function}. ',
                'Expected one of "affine", "scale", "translate", "identity".',
            )
        self.normalize_function = normalize_function
        self.register_buffer('mean', torch.zeros(shape, device=device))
        self.register_buffer('var', torch.ones(shape, device=device))
        self.register_buffer('count', torch.zeros(1, dtype=torch.long, device=device))

    @abstractmethod
    def update(self, data: torch.Tensor) -> None:
        """Update mean and variance."""
        raise NotImplementedError

    @property
    def std(self) -> torch.Tensor:
        """Return standard deviation."""
        return self.var.sqrt()

    def set_mean_var(
        self,
        mean: torch.Tensor | list[float] | tuple[float, ...] | None,
        var: torch.Tensor | list[float] | tuple[float, ...] | None,
    ) -> None:
        """Set mean and variance."""
        mean = (
            torch.as_tensor(mean, dtype=self.mean.dtype, device=self.mean.device)
            if mean is not None
            else self.mean
        )
        var = (
            torch.as_tensor(var, dtype=self.var.dtype, device=self.var.device)
            if var is not None
            else self.var
        )

        assert mean.shape == self.mean.shape
        assert var.shape == self.var.shape

        self.mean = mean
        self.var = var

    def forward(
        self,
        data: torch.Tensor,
        epsilon: Number = 1e-8,
    ) -> torch.Tensor:
        """Update and normalize input."""
        if self.training:
            self.update(data)
        return self.normalize(data, epsilon=epsilon)

    def normalize(
        self,
        data: torch.Tensor,
        epsilon: Number = 1e-8,
    ) -> torch.Tensor:
        """Normalize input."""
        if self.normalize_function == 'affine':
            return (data - self.mean.detach()) / (self.std.detach() + epsilon)
        if self.normalize_function == 'scale':
            return data / (self.std.detach() + epsilon)
        if self.normalize_function == 'translate':
            return data - self.mean.detach()
        if self.normalize_function == 'identity':
            return data
        raise ValueError(
            f'Invalid normalization function type: {self.normalize_function}. ',
            'Expected one of "affine", "scale", "translate", "identity".',
        )

    @classmethod
    def instantiate(
        cls,
        normalizer_type: NormalizerType | None,
        normalize_function: NormalizeFunction,
        shape: tuple[int, ...],
        device: torch.device | str | None = None,
        **kwargs: Any,
    ) -> Normalizer:
        """Get a normalizer."""
        if normalizer_type == 'RunningMeanStd':
            return RunningMeanStd(
                normalize_function,
                shape=shape,
                device=device,
            )
        if normalizer_type == 'ExponentialMovingAverage':
            return ExponentialMovingAverage(
                normalize_function,
                shape=shape,
                device=device,
                **kwargs,
            )
        if normalizer_type is None:
            return IdentityNormalizer(
                normalize_function,
                shape=shape,
                device=device,
            )
        raise ValueError(
            f'Invalid normalization function type: {normalizer_type}. '
            'Expected one of "RunningMeanStd", "ExponentialMovingAverage".',
        )


class RunningMeanStd(Normalizer):
    """Running mean and standard deviation."""

    def update(self, data: torch.Tensor) -> None:
        """Update mean and variance."""
        batch_mean = data.mean(dim=0)
        batch_var = data.var(dim=0)
        batch_count = data.size(0)

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = (  # pylint: disable=invalid-name
            m_a + m_b + torch.square(delta) * (self.count * batch_count / total_count)
        )
        new_var = m2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count


class ExponentialMovingAverage(Normalizer):
    """Exponential moving average."""

    def __init__(
        self,
        normalize_function: NormalizeFunction,
        shape: tuple[int, ...],
        device: torch.device | str | None = None,
        momentum: float = 0.9,
    ) -> None:
        super().__init__(normalize_function, shape=shape, device=device)
        self.momentum = momentum

    def update(self, data: torch.Tensor) -> None:
        """Update mean and variance."""
        batch_mean = data.mean(dim=0)
        batch_var = data.var(dim=0)
        batch_count = data.size(0)

        self.mean = self.momentum * self.mean + (1.0 - self.momentum) * batch_mean
        self.var = self.momentum * self.var + (1.0 - self.momentum) * batch_var
        self.count += batch_count  # pylint: disable=no-member


class IdentityNormalizer(Normalizer):
    """Identity normalizer."""

    def update(self, data: torch.Tensor) -> None:
        """Update mean and variance."""
        self.count += data.size(0)  # pylint: disable=no-member



class ScoreModelMixin:
    """Base class for score models."""

    score_head: nn.Linear
    normalizer: Normalizer
    do_normalize: bool = False
    normalize_function: NormalizeFunction = 'affine'
    _initialized: bool = False

    def init_score_head(self, config: PretrainedConfig, hidden_size: int, **kwargs: Any) -> None:
        """Initialize the score head."""
        if self._initialized:
            return

        config.score_dim = kwargs.pop('score_dim', getattr(config, 'score_dim', 1))
        config.bias = kwargs.pop('bias', getattr(config, 'bias', False))

        config.score_type = kwargs.pop('score_type', getattr(config, 'score_type', 'reward'))
        if config.score_type == 'reward':
            self.normalize_function = 'affine'
        elif config.score_type == 'cost':
            self.normalize_function = 'scale'
        elif config.score_type == 'critic':
            self.normalize_function = 'identity'
        else:
            raise ValueError(
                f"Invalid score type: {config.score_type}. Expected one of 'reward', 'cost', or 'critic'.",
            )
        config.do_normalize = kwargs.pop(
            'do_normalize',
            getattr(config, 'do_normalize', False),
        )
        self.do_normalize = config.do_normalize

        config.normalizer_type = kwargs.pop(
            'normalizer_type',
            getattr(config, 'normalizer_type', None),
        )
        if config.normalizer_type not in {'RunningMeanStd', 'ExponentialMovingAverage', None}:
            raise ValueError(
                f'Invalid norm type: {config.normalizer_type}.'
                "Expected one of 'RunningMeadStd', 'ExponentialMovingAverage', or None.",
            )
        if config.normalizer_type == 'ExponentialMovingAverage':
            config.momentum = kwargs.pop('momentum', getattr(config, 'momentum', None))
        momentum = getattr(config, 'momentum', None)

        self.score_head = nn.Linear(hidden_size, config.score_dim, bias=config.bias)
        self.normalizer = Normalizer.instantiate(
            normalizer_type=config.normalizer_type,
            normalize_function=self.normalize_function,
            shape=(config.score_dim,),
            momentum=momentum,
        )

        mean = getattr(config, 'mean', None)
        var = getattr(config, 'var', None)
        self.normalizer.set_mean_var(mean, var)

        self._initialized = True

    def get_score(
        self,
        hidden_state: torch.Tensor,  # size = (B, L, E)
        attention_mask: torch.BoolTensor,  # size = (B, L)
        return_dict: bool | None = None,
    ) -> ScoreModelOutput:
        """Forward pass of the score model."""
        scores = self.score_head(hidden_state)  # size = (B, L, D)

        end_score = []
        for i in range(hidden_state.size(0)):
            end_index = attention_mask[i].nonzero()[-1].item()
            end_score.append(scores[i, end_index])  # size = (D,)
        end_score = torch.stack(end_score, dim=0)  # size = (B, D)

        if self.training:
            if dist.is_initialized():
                gathered_end_score_list = [
                    torch.zeros_like(end_score) for _ in range(dist.get_world_size())
                ]
                dist.all_gather(gathered_end_score_list, end_score)
                gathered_end_score = torch.cat(gathered_end_score_list, dim=0)
                self.normalizer.update(gathered_end_score)
            else:
                self.normalizer.update(end_score)
            self.config.mean = self.normalizer.mean.tolist()
            self.config.var = self.normalizer.var.tolist()

        if self.do_normalize:
            scores = self.normalizer.normalize(scores)
            end_score = self.normalizer.normalize(end_score)

        if not return_dict:
            return scores, end_score

        return ScoreModelOutput(
            scores=scores,  # size = (B, L, D)
            end_scores=end_score,  # size = (B, D)
        )

    def set_normalize(self, mode: bool = True) -> None:
        if self.do_normalize == mode:
            return

        self.do_normalize = self.config.do_normalize = mode


class LlamaModelForScore(ScoreModelMixin, LlamaPreTrainedModel):
    _keys_to_ignore_on_load_missing: ClassVar[list[str]] = ['lm_head.weight']

    def __init__(self, config: PretrainedConfig, **kwargs: Any) -> None:
        print("LlamaModelForScore init")
        super().__init__(config)
        self.model = LlamaModel(config)

        config.architectures = [self.__class__.__name__]
        self.init_score_head(config, hidden_size=config.hidden_size, **kwargs)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> None:
        return None

    def set_decoder(self, decoder: PreTrainedModel) -> None:
        self.model = decoder

    def get_decoder(self) -> PreTrainedModel:
        return self.model

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=ScoreModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(  # pylint: disable=too-many-arguments
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        labels: torch.LongTensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | ScoreModelOutput | ScoreModelOutputWithPastkeyvalues:
        """
        Args:

        Returns:

        Examples:

        ```python
        >>> from safe_rlhf.models.llama.modeling_llama import LlamaModelForScore
        >>> from transformers import LlamaTokenizer

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        # got score
        >>> outputs = model(**inputs)
        >>> scores = outputs.scores
        >>> scores
        tensor([[[0.0000]]])
        ```
        """
        assert attention_mask is not None
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # breakpoint()
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]  # size = (B, L, E)

        if attention_mask.shape[1] != hidden_states.shape[1]:
            scores = self.score_head(hidden_states) 
            return ScoreModelOutputWithPastkeyvalues(
                scores=scores,
                end_scores=scores[:, -1, :],
                past_key_values=outputs.past_key_values,
            )


        score_model_output = self.get_score(
            hidden_states,
            attention_mask=attention_mask,
            return_dict=return_dict,
        )
        return ScoreModelOutputWithPastkeyvalues(
            scores=score_model_output.scores,
            end_scores=score_model_output.end_scores,
            past_key_values=outputs.past_key_values,
        ) 


if __name__ == "__main__":
    model = LlamaModelForScore.from_pretrained("/mnt/petrelfs/share_data/llm-safety/models/golden_rm_summarize").cuda()
    tokenizer = AutoTokenizer.from_pretrained("/mnt/petrelfs/share_data/llm-safety/models/golden_rm_summarize")
    from datasets import load_dataset
    dataset = load_dataset("CarperAI/openai_summarize_comparisons")
    acc = []
    from tqdm import tqdm
    for i in tqdm(range(1, 100)):
        prompt = dataset["train"][i]["prompt"]
        chosen = dataset["train"][i]["chosen"]
        rejected = dataset["train"][i]["rejected"]
        breakpoint()
        chosen_text = prompt + chosen + tokenizer.eos_token
        rejected_text = prompt + rejected + tokenizer.eos_token
        inputs = tokenizer([chosen_text, rejected_text], return_tensors="pt", padding=True)
        inputs = {k: v.cuda() for k, v in inputs.items()}
        outputs = model(**inputs)
        # print(outputs.end_scores)
        scores = outputs.end_scores.squeeze(0)
        acc.append(scores[0] > scores[1])
    print(sum(acc) / len(acc))