# -*- coding: utf-8 -*-
from torch import Tensor, nn


class TimeLayerNorm(nn.LayerNorm):
    # pylint: disable=arguments-renamed
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.permute(0, 2, 1)).permute(0, 2, 1)
