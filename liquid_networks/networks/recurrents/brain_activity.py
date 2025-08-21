import torch as th
from torch.nn import functional as th_f

from .simple import LiquidRecurrent


class BrainActivityLiquidRecurrent(LiquidRecurrent):
    def _output_processing(self, out: th.Tensor) -> th.Tensor:
        return out

    def _sequence_processing(self, outputs: list[th.Tensor]) -> th.Tensor:
        return th_f.softmax(
            super()._output_processing(th.stack(outputs, dim=1).mean(dim=1)),
            dim=-1,
        )
