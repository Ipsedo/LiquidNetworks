# -*- coding: utf-8 -*-
from pathlib import Path
from typing import NamedTuple, Union

from .networks import LiquidRecurrent


class ModelOptions(NamedTuple):
    neuron_number: int
    input_size: int
    unfolding_steps: int
    output_size: int

    def get_model(self) -> LiquidRecurrent:
        return LiquidRecurrent(
            self.neuron_number,
            self.input_size,
            self.unfolding_steps,
            self.output_size,
        )


class TrainOptions(NamedTuple):
    epoch: int
    batch_size: int
    learning_rate: float
    output_folder: Union[str, Path]
    run_name: str
    metric_window_size: int
    cuda: bool
