from pathlib import Path
from typing import Any, Callable, NamedTuple

import torch as th

from .data import AbstractDataset, DatasetNames, get_dataset_constructor
from .networks import (
    AbstractLiquidRecurrent,
    ActivationFunction,
    TaskType,
    get_activation_fn,
    get_loss_function,
    get_model_constructor,
)
from .networks.functions import ReductionType


class ModelOptions(NamedTuple):
    neuron_number: int
    unfolding_steps: int
    activation_function: ActivationFunction
    task_type: TaskType
    specific_parameters: dict[str, str]
    cuda: bool

    def get_model(self) -> AbstractLiquidRecurrent:
        return get_model_constructor(self.task_type)(
            self.specific_parameters
        ).get_recurrent(
            self.neuron_number,
            self.unfolding_steps,
            get_activation_fn(self.activation_function),
        )

    def get_loss_function(
        self,
    ) -> Callable[[th.Tensor, th.Tensor, ReductionType], th.Tensor]:
        return get_loss_function(self.task_type)

    def to_dict(self) -> dict[str, Any]:
        return dict(self._asdict())  # pylint: disable=no-member


class TrainOptions(NamedTuple):
    epoch: int
    batch_size: int
    learning_rate: float
    output_folder: str | Path
    run_name: str
    metric_window_size: int
    dataset_name: DatasetNames
    train_data_path: str
    valid_data_path: str | None
    save_every: int
    eval_every: int

    def to_dict(self) -> dict[str, Any]:
        return dict(self._asdict())  # pylint: disable=no-member

    def get_train_dataset(self) -> AbstractDataset:
        return get_dataset_constructor(self.dataset_name)(self.train_data_path)

    def get_valid_dataset(self) -> AbstractDataset | None:
        if self.valid_data_path is None:
            return None
        return get_dataset_constructor(self.dataset_name)(self.valid_data_path)
