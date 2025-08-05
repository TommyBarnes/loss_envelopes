from abc import ABC
from typing import Dict, Type
import torch
from torch import Tensor

from .unbiased_ema import UnbiasedEMA
from .envelope_shape import EnvelopeShape

class LossEnvelopeBase(ABC):
    def __init__(self, method: Type[EnvelopeShape], static_parameters: Dict[str, float]) -> None:
        self.method = method
        self._static_parameters = static_parameters
        self._logging_dict = {}

    def _log_weight(self, weights: Tensor) -> None:
        self._logging_dict['avg_weight'] = weights.mean().item()

    def logging_dict(self) -> Dict:
        return self._logging_dict

class StaticLossEnvelope(LossEnvelopeBase):
    def call_for_weight(self, loss: Tensor):
        weight = self.method.compute_weight(loss, self._static_parameters)
        self._log_weight(weight)
        return weight.detach()

    def __call__(self, loss: Tensor) -> Tensor:
        weight = self.method.compute_weight(loss, self._static_parameters)
        self._log_weight(weight)
        return loss * weight.detach()

class AdaptiveLossEnvelopeBase(LossEnvelopeBase):
    def __init__(self, method: Type[EnvelopeShape], static_parameters: Dict[str, float], adaptive_parameters: Dict[str, float], step_size: float ) -> None:
        super().__init__(method, static_parameters)
        self._adaptive_parameters = adaptive_parameters

        self.parameter_emas = {
            k: UnbiasedEMA(step_size=step_size)
            for k in self._adaptive_parameters.keys()
        }

    def _parameters_from_values(self, values: Tensor):
        # Start with static parameters
        parameters = dict(self._static_parameters)
        for k in self._adaptive_parameters.keys():
            if k == 'invslope':
                # Compute invslope span based on quantile values around the center
                center_q = self._adaptive_parameters['T']
                span_pct = self._adaptive_parameters['invslope']
                q_min = max(0.0, center_q - 0.5 * span_pct)
                q_max = min(1.0, center_q + 0.5 * span_pct)
                q_min_val = torch.quantile(values, q_min).item()
                q_max_val = torch.quantile(values, q_max).item()
                span = q_max_val - q_min_val + 1e-8
                parameters['invslope'] = self.parameter_emas['invslope'].update(span)
            else:
                q = self._adaptive_parameters[k]
                parameters[k] = self.parameter_emas[k].update(torch.quantile(values, q).item())  # type: ignore
        return parameters
    
    def logging_dict(self) -> Dict:
        logging_dict = super().logging_dict()
        # Only include EMA values for adaptive parameters
        for k, ema in self.parameter_emas.items():
            logging_dict[k] = ema.value
        return logging_dict

class AdaptiveLossEnvelope(AdaptiveLossEnvelopeBase):
    def call_for_weight(self, loss: Tensor):
        parameters = self._parameters_from_values(loss)
        weight = self.method.compute_weight(loss, parameters)
        self._log_weight(weight)
        return weight.detach()

    def __call__(self, loss: Tensor) -> Tensor:
        parameters = self._parameters_from_values(loss)
        weight = self.method.compute_weight(loss, parameters)
        self._log_weight(weight)
        return loss * weight.detach()

    
