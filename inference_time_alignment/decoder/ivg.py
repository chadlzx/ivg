from dataclasses import dataclass
import warnings
from typing import Dict, List, Any, Optional, Union

import torch
from torch import nn
from transformers import PreTrainedModel, GenerationMixin

__all__ = ['TokenWiseSamplingMixin']
@dataclass(kw_only=True)
class TokenWiseSamplingMixin(GenerationMixin):
    base: PreTrainedModel
    token_reward_model: PreTrainedModel
    w: float

    def __getattribute__(self, name: str) -> Any:
        try:
            return super().__getattribute__(name)
        except AttributeError:
            return getattr(self.base, name)

    def prepare_inputs_for_generation(self, input_ids, **model_kwargs):
        if 'past_key_values' in model_kwargs:
            past_key_values = model_kwargs['past_key_values']
            model_kwargs['past_key_values'] = model_kwargs['past_key_values']['base']
        result = self.base.prepare_inputs_for_generation(input_ids, **model_kwargs)
        if 'past_key_values' in model_kwargs:
            result['past_key_values'] = past_key_values
        return result

    @torch.no_grad()
    def __call__(self, *args, past_key_values=None, **kwargs):
        if not past_key_values:
            past_key_values = {'base': None, 'token_reward': None}

        base_outputs = self.base(*args, past_key_values=past_key_values['base'], **kwargs)
        base_logits  = base_outputs.logits[:, -1, :]

        token_reward_outputs = self.token_reward_model(*args, past_key_values=past_key_values['token_reward'], **kwargs)
        token_reward_logits  = token_reward_outputs.logits[:, -1, :]

        r = self.w * token_reward_logits
        logits = base_logits + r

        outputs = base_outputs
        outputs.logits = logits.unsqueeze(-2)
        outputs.past_key_values = {
            'base':          base_outputs.past_key_values,
            'token_reward':  token_reward_outputs.past_key_values,
        }
        return outputs



