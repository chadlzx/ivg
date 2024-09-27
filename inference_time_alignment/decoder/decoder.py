"""
This module consolidates various mixin classes used in the inference-time alignment process for different decoding strategies.

The `__all__` list defines the public interface of this module, making only the listed classes available when importing this module using `from <module> import *`.
"""


from inference_time_alignment.decoder.ivg import TokenWiseSamplingMixin
from inference_time_alignment.decoder.cbs import BeamTuningWithEFTPosthocGenerationMixin
from inference_time_alignment.decoder.eft import EFTPosthocGenerationMixin

__all__ = [
    "EFTPosthocGenerationMixin",
    "TokenWiseSamplingMixin",
    "BeamTuningWithEFTPosthocGenerationMixin",
]
