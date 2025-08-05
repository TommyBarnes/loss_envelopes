from pydantic import BaseModel, Field
from typing import List, Optional, Union, Dict, Callable

from .loss_envelopes import StaticLossEnvelope, AdaptiveLossEnvelope
from .envelope_shape import get_envelope_shape

# Static parameter is a float
StaticParam = float

class LossAdaptiveParam(BaseModel):
    loss_pct: float = Field(..., description="Percentile in loss space to set parameter")

ParamSpec = Union[StaticParam, LossAdaptiveParam]

class EnvelopeConfig(BaseModel):
    shape: str  # e.g., "identity", "step", "linear", "sigmoid"
    parameters: Optional[Dict[str, ParamSpec]] = None
    step_size: float = 0.05

    @property
    def adaptivity(self)->str:
        loss_adaptive = self.is_loss_adaptive
        if loss_adaptive:
            return 'loss_adaptive'
        return 'static'

    @property
    def is_loss_adaptive(self):
        if not self.parameters:
            return False
        return any(isinstance(v, LossAdaptiveParam) for v in self.parameters.values())
        
    @property
    def is_static(self):
        return not self.is_loss_adaptive

    def to_slug(self, exclude: Optional[List[str]] = None) -> str:
        if exclude is None:
            exclude = []
        if not self.parameters:
            return self.shape
        parts = []
        for key in sorted(self.parameters.keys()):
            if key in exclude:
                continue
            val = self.parameters[key]
            if isinstance(val, (float, int)):
                parts.append(f"{key}{val}")
            elif isinstance(val, LossAdaptiveParam):
                parts.append(f"{key}V{val.loss_pct}")
            else:
                parts.append(f"{key}{val}")
        if not self.is_static and 'ss' not in exclude:
            parts.append(f"ss{self.step_size}")
        return self.shape + "-" + "-".join(parts)

    @classmethod
    def from_slug(cls, slug: str) -> "EnvelopeConfig":
        """
        Parse a slug like 'step-center0.5-width0.1' or 'identity' into an EnvelopeConfig.
        """
        parts = slug.split('-')
        shape = parts[0]
        # Initialize with no parameters by default
        params: Dict[str, ParamSpec] = {}
        step_size = cls.model_fields['step_size'].get_default()
        # If only the shape name is present, return with defaults
        if len(parts) == 1:
            return cls(shape=shape, parameters=None)

        for token in parts[1:]:
            if token.startswith('ss'):
                step_size = float(token[2:])
            # loss-adaptive token: contains 'V' after key, e.g. "centerT0.5"
            elif 'V' in token and not token.startswith(('V',)):
                key, rest = token.split('V', 1)
                params[key] = LossAdaptiveParam(loss_pct=float(rest))
            else:
                # static numeric parameter: split into key and numeric value
                idx = 0
                while idx < len(token) and not (token[idx].isdigit() or token[idx] == '.'):
                    idx += 1
                key = token[:idx]
                val_str = token[idx:]
                # parse as float if contains '.', else int
                val: Union[int, float]
                if '.' in val_str:
                    val = float(val_str)
                else:
                    val = int(val_str)
                params[key] = val

        return cls(shape=shape, parameters=params if params else None, step_size=step_size)


def build_loss_envelope(config: EnvelopeConfig, loss_fn: Optional[Callable] = None):
    shape_cls = get_envelope_shape(config.shape)
    param_dict = config.parameters or {}

    static_params = {
        k: float(v) for k, v in param_dict.items() if isinstance(v, (float, int))
    }
    adaptive_params = {}
    for k, v in param_dict.items():
        if isinstance(v, LossAdaptiveParam):
            adaptive_params[k] = v.loss_pct

    uses_loss_adaptive = any(isinstance(v, LossAdaptiveParam) for v in param_dict.values())

    if uses_loss_adaptive:
        return AdaptiveLossEnvelope(
            shape_cls,
            static_params,
            adaptive_params,
            step_size=config.step_size
        )

    else:
        return StaticLossEnvelope(shape_cls, static_params)
