# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Any, Dict, NamedTuple, Union

from .networks import LiquidRecurrent


class ModelOptions(NamedTuple):
    neuron_number: int
    input_size: int
    unfolding_steps: int
    output_size: int
    cuda: bool

    def get_model(self) -> LiquidRecurrent:
        return LiquidRecurrent(
            self.neuron_number,
            self.input_size,
            self.unfolding_steps,
            self.output_size,
        )

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._asdict())  # pylint: disable=no-member


class TrainOptions(NamedTuple):
    epoch: int
    batch_size: int
    learning_rate: float
    output_folder: Union[str, Path]
    run_name: str
    metric_window_size: int
    csv_path: str

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._asdict())  # pylint: disable=no-member
