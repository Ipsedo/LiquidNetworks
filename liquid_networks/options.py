from pathlib import Path
from typing import Callable

import torch as th
from pydantic import BaseModel

from .data import AbstractDataset, DatasetNames, get_dataset_constructor
from .networks import (
    AbstractLiquidRecurrent,
    ActivationFunction,
    TaskType,
    get_activation_fn,
    get_loss_function,
    get_model_constructor,
)
from .networks.losses import ReductionType


class ModelOptions(BaseModel):
    neuron_number: int
    unfolding_steps: int
    activation_function: ActivationFunction
    delta_t: float
    task_type: TaskType
    specific_parameters: dict[str, str]
    cuda: bool

    def get_model(self) -> AbstractLiquidRecurrent:
        return get_model_constructor(self.task_type)(self.specific_parameters).get_recurrent(
            self.neuron_number,
            self.unfolding_steps,
            get_activation_fn(self.activation_function),
            self.delta_t,
        )

    def get_loss_function(
        self,
    ) -> Callable[[th.Tensor, th.Tensor, ReductionType], th.Tensor]:
        return get_loss_function(self.task_type)

    def get_device(self) -> th.device:
        if self.cuda:
            th.backends.cudnn.benchmark = True
            return th.device("cuda")

        return th.device("cpu")


class TrainOptions(BaseModel):
    seed: int
    epoch: int
    batch_size: int
    learning_rate: float
    output_folder: str | Path
    run_name: str
    metric_window_size: int
    dataset_name: DatasetNames
    dataset_parameters: dict[str, str]
    train_dataset_path: str
    valid_dataset_path: str | None
    save_every: int
    eval_every: int
    workers: int

    def get_train_dataset(self) -> AbstractDataset:
        return get_dataset_constructor(self.dataset_name)(self.dataset_parameters).get_dataset(
            self.train_dataset_path
        )

    def get_valid_dataset(self) -> AbstractDataset | None:
        if self.valid_dataset_path is None:
            return None
        return get_dataset_constructor(self.dataset_name)(self.dataset_parameters).get_dataset(
            self.valid_dataset_path
        )


class EvalOptions(BaseModel):
    model_path: str
    output_folder: str
    run_name: str
    dataset_name: DatasetNames
    dataset_parameters: dict[str, str]
    dataset_path: str
    batch_size: int
    workers: int

    def get_dataset(self) -> AbstractDataset:
        return get_dataset_constructor(self.dataset_name)(self.dataset_parameters).get_dataset(
            self.dataset_path
        )
