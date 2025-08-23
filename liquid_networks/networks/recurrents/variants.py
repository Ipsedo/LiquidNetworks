import torch as th
from torch.nn import functional as th_f

from .simple import BaseLiquidRecurrentFactory, LiquidRecurrent


# Last time steps
class LastLiquidRecurrent(LiquidRecurrent):
    def _sequence_processing(self, outputs: list[th.Tensor]) -> th.Tensor:
        return outputs[-1]


class LastLiquidRecurrentFactory(
    BaseLiquidRecurrentFactory[LastLiquidRecurrent]
):
    def __init__(self, config: dict[str, str]) -> None:
        super().__init__(config, LastLiquidRecurrent)


# softmax output
class SoftmaxLiquidRecurrent(LiquidRecurrent):
    def _output_processing(self, out: th.Tensor) -> th.Tensor:
        return th_f.softmax(super()._output_processing(out), -1)


class SoftmaxLiquidRecurrentFactory(
    BaseLiquidRecurrentFactory[SoftmaxLiquidRecurrent]
):
    def __init__(self, config: dict[str, str]) -> None:
        super().__init__(config, SoftmaxLiquidRecurrent)


# softmax output last time step
class LastSoftmaxLiquidRecurrent(SoftmaxLiquidRecurrent):
    def _sequence_processing(self, outputs: list[th.Tensor]) -> th.Tensor:
        return outputs[-1]


class LastSoftmaxLiquidRecurrentFactory(
    BaseLiquidRecurrentFactory[LastSoftmaxLiquidRecurrent]
):
    def __init__(self, config: dict[str, str]) -> None:
        super().__init__(config, LastSoftmaxLiquidRecurrent)


# sigmoid
class SigmoidLiquidRecurrent(LiquidRecurrent):
    def _output_processing(self, out: th.Tensor) -> th.Tensor:
        return th_f.sigmoid(super()._output_processing(out))


class SigmoidLiquidRecurrentFactory(
    BaseLiquidRecurrentFactory[SigmoidLiquidRecurrent]
):
    def __init__(self, config: dict[str, str]) -> None:
        super().__init__(config, SigmoidLiquidRecurrent)


# softplus
class SoftplusLiquidRecurrent(LiquidRecurrent):
    def _output_processing(self, out: th.Tensor) -> th.Tensor:
        # pylint: disable=not-callable
        return th_f.softplus(super()._output_processing(out))


class SoftplusLiquidRecurrentFactory(
    BaseLiquidRecurrentFactory[SoftplusLiquidRecurrent]
):
    def __init__(self, config: dict[str, str]) -> None:
        super().__init__(config, SoftplusLiquidRecurrent)
