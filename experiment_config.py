# experiment_config.py
import os
import yaml
from pydantic import BaseModel, Field, ValidationError
from typing import Optional, List

from hyperparameter_config import HParamSweep

class DriftConfig(BaseModel):
    enabled: bool = True    # only used as a sentinel to instantiate drift config with default values. not checked
    phase_spread: float = Field(default=1.0, ge=0.0, le=1.0)
    shift_spread: float = Field(default=1.0, ge=0.0, le=1.0)
    frequency_std: float = Field(default=1.0, ge=0.0)
    frequency: float = Field(default=0.2, ge=0.0)  # base frequency
    shift: float = Field(default=1.0)              # baseline shift
    amplitude: float = Field(default=1.0)          # sine amplitude
    frequency_per_class: bool = False              # if True, classes will not have the same frequency

class RLConfig(BaseModel):
    environment: str
    max_steps: int
    rollout_size: Optional[int]
    ppo_epochs: Optional[int]
    minibatch_size: Optional[int]


class DatasetConfig(BaseModel):
    train_start: int
    train_size: int
    train_batch_size: int
    val_start: int
    val_size: int
    val_batch_size: int
    drift: Optional[DriftConfig] = None

class ClassifierHeadConfig(BaseModel):
    enabled: bool
    type: str
    hidden_dim: int

class EvaluationConfig(BaseModel):
    classifier_head: Optional[ClassifierHeadConfig]

class ExperimentSchema(BaseModel):
    name: str
    rl: Optional[RLConfig] = None
    dataset: Optional[DatasetConfig] = None
    evaluation: Optional[EvaluationConfig] = None
    epochs: Optional[int] = None
    sweep: HParamSweep

class ExperimentConfig(ExperimentSchema):
    def __init__(self, path):
        with open(path, 'r') as f:
            raw = yaml.safe_load(f)
        try:
            super().__init__(**raw)
        except ValidationError as e:
            raise ValueError(f"Experiment config validation failed: {e}")

    def save_to(self, directory):
        os.makedirs(directory, exist_ok=True)
        target_path = os.path.join(directory, "experiment_config.yaml")

        if os.path.exists(target_path):
            with open(target_path, 'r') as f:
                existing = yaml.safe_load(f)
            same = [v==existing[k] for k,v in self.model_dump().items() if k != 'sweep']
            if not all(same):
                raise ValueError(f"Experiment config mismatch in {directory}/experiment_config.yaml")

        else:
            with open(target_path, 'w') as f:
                yaml.safe_dump(self.model_dump(), f)