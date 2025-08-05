from abc import ABC, abstractmethod
from typing import List, Dict, Any, Type
import torch
from torch import Tensor

class EnvelopeShape(ABC):
    @classmethod
    @abstractmethod
    def parameters(cls) -> List[str]:
        pass

    @classmethod
    @abstractmethod
    def compute_weight(cls, values: Tensor, parameters: Dict[str, Any]) -> Tensor:
        pass
        
class IdentityShape(EnvelopeShape):
    @classmethod
    def parameters(cls) -> List[str]:
        return []

    @classmethod
    def compute_weight(cls, values: Tensor, parameters: Dict[str, Any]) -> Tensor:
        return torch.ones_like(values)

class StepShape(EnvelopeShape):
    @classmethod
    def parameters(cls) -> List[str]:
        return ['T']

    @classmethod
    def compute_weight(cls, values: Tensor, parameters: Dict[str, Any]) -> Tensor:
        threshold = parameters['T']
        return (values > threshold).float()

class LinearShape(EnvelopeShape):
    @classmethod
    def parameters(cls) -> List[str]:
        return ['T', 'invslope']

    @classmethod
    def compute_weight(cls, values: Tensor, parameters: Dict[str, Any]) -> Tensor:
        slope = 1.0 / parameters['invslope']
        half_width = 0.5 / slope
        start = parameters['T'] - half_width
        end = parameters['T'] + half_width
        span = end - start if end > start else 1e-8
        w = (values - start) / span
        return w.clamp(min=0.0, max=1.0)

class SigmoidShape(EnvelopeShape):
    @classmethod
    def parameters(cls) -> List[str]:
        return ['T', 'invslope']

    @classmethod
    def compute_weight(cls, values: Tensor, parameters: Dict[str, Any]) -> Tensor:
        threshold = parameters['T']
        slope = 1.0 / parameters['invslope']
        beta = 4 * slope
        return torch.sigmoid(beta * (values - threshold))

# Factory method to get envelope shape class by name
def get_envelope_shape(name: str) -> Type[EnvelopeShape]:
    shape_map = {
        'identity': IdentityShape,
        'step': StepShape,
        'linear': LinearShape,
        'sigmoid': SigmoidShape,
    }
    if name not in shape_map:
        raise ValueError(f"Unknown envelope shape name: {name}")
    return shape_map[name]
