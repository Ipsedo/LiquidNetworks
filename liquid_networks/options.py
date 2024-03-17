# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Any, Callable, Dict, NamedTuple, Optional, Union

import torch as th

from .data import AbstractDataset, DatasetNames, get_dataset_constructor
from .networks import (
    LiquidRecurrent,
    ReductionType,
    TaskType,
    get_loss_function,
    get_model_constructor,
)


class ModelOptions(NamedTuple):
    neuron_number: int
    input_size: int
    unfolding_steps: int
    output_size: int
    task_type: TaskType
    cuda: bool

    def get_model(self) -> LiquidRecurrent:
        return get_model_constructor(self.task_type)(
            self.neuron_number,
            self.input_size,
            self.unfolding_steps,
            self.output_size,
        )

    def get_loss_function(
        self,
    ) -> Callable[[th.Tensor, th.Tensor, ReductionType], th.Tensor]:
        return get_loss_function(self.task_type)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._asdict())  # pylint: disable=no-member


class TrainOptions(NamedTuple):
    epoch: int
    batch_size: int
    learning_rate: float
    output_folder: Union[str, Path]
    run_name: str
    metric_window_size: int
    dataset_name: DatasetNames
    train_data_path: str
    valid_data_path: Optional[str]
    save_every: int
    eval_every: int

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._asdict())  # pylint: disable=no-member

    def get_train_dataset(self) -> AbstractDataset:
        return get_dataset_constructor(self.dataset_name)(self.train_data_path)

    def get_valid_dataset(self) -> Optional[AbstractDataset]:
        if self.valid_data_path is None:
            return None
        return get_dataset_constructor(self.dataset_name)(self.valid_data_path)
