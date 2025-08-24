from os import makedirs
from os.path import exists, isdir, join
from pathlib import Path

import torch as th
from torch import nn


class ModelSaver:
    def __init__(
        self,
        prefix_name: str,
        output_folder: str | Path,
        module: nn.Module,
        optim: th.optim.Optimizer,
        save_every: int,
    ) -> None:
        self.__prefix = prefix_name

        self.__output_folder = output_folder

        self.__module = module
        self.__optim = optim

        self.__save_every = save_every

        self.__curr_idx = 0
        self.__global_counter = 0

    def tick_save(self) -> None:
        if self.__curr_idx % self.__save_every == 0:
            if not exists(self.__output_folder):
                makedirs(self.__output_folder)
            elif not isdir(self.__output_folder):
                raise NotADirectoryError(
                    f"Output folder '{self.__output_folder}' is not a directory."
                )

            th.save(
                self.__module.state_dict(),
                join(self.__output_folder, f"{self.__prefix}_{self.__global_counter}.pt"),
            )

            th.save(
                self.__optim.state_dict(),
                join(self.__output_folder, f"{self.__prefix}_{self.__global_counter}_optim.pt"),
            )

        self.__curr_idx = (self.__curr_idx + 1) % self.__save_every
        self.__global_counter += 1
