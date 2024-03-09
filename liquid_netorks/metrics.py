# -*- coding: utf-8 -*-
from statistics import mean
from typing import List, Union

import torch as th


class Metric:
    def __init__(self, window_size: int) -> None:
        self.__window_size = window_size
        self.__result: List[float] = [0.0]

    def add_result(self, res: Union[th.Tensor, float]) -> None:
        if isinstance(res, th.Tensor):
            res = res.mean()
            res_float = res.item()
        else:
            res_float = res

        self.__result.append(res_float)

        while len(self.__result) > self.__window_size:
            self.__result.pop(0)

    def get_smoothed_metric(self) -> float:
        return mean(self.__result)

    def get_last_metric(self) -> float:
        return self.__result[-1]
