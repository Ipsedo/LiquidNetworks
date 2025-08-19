import torch as th
from torch.nn import functional as th_f

from .simple import LiquidRecurrent


class LastLiquidRecurrent(LiquidRecurrent):
    def _sequence_processing(self, outputs: list[th.Tensor]) -> th.Tensor:
        return outputs[-1]


class SoftmaxLiquidRecurrent(LiquidRecurrent):
    def _output_processing(self, out: th.Tensor) -> th.Tensor:
        return th_f.softmax(super()._output_processing(out), -1)


class LastSoftmaxLiquidRecurrent(SoftmaxLiquidRecurrent):
    def _sequence_processing(self, outputs: list[th.Tensor]) -> th.Tensor:
        return outputs[-1]


class SigmoidLiquidRecurrent(LiquidRecurrent):
    def _output_processing(self, out: th.Tensor) -> th.Tensor:
        return th_f.sigmoid(super()._output_processing(out))


class SoftplusLiquidRecurrent(LiquidRecurrent):
    def _output_processing(self, out: th.Tensor) -> th.Tensor:
        # pylint: disable=not-callable
        return th_f.softplus(super()._output_processing(out))
