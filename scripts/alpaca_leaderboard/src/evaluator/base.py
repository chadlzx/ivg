from abc import ABC
from dataclasses import dataclass
from typing import Union, Text, List


@dataclass
class EvaluationInput:
    prompt: Text
    response: Text


class BaseEvaluator(ABC):

    def eval(self, input: Union[EvaluationInput, List[EvaluationInput]]) -> List[float]:
        raise NotImplementedError
