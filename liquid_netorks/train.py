# -*- coding: utf-8 -*-
from .options import ModelOptions, TrainOptions


def train(model_options: ModelOptions, train_options: TrainOptions) -> None:
    print(model_options)
    print(train_options)
