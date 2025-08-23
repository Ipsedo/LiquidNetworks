from typing import Callable

import torch as th
from torch.utils.data import DataLoader

from .data import AbstractDataset
from .networks import AbstractLiquidRecurrent
from .networks.functions import LossFunctionType


def eval_model_on_dataset[T](
    ltc: AbstractLiquidRecurrent[T],
    valid_dataset: AbstractDataset[T],
    batch_size: int,
    device: th.device,
    loss_fn: LossFunctionType,
    callback_batch_iter: Callable[[int, int], None] | None = None,
) -> float:
    with th.no_grad():
        ltc.eval()

        valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            num_workers=6,
            collate_fn=valid_dataset.collate_fn,
        )

        valid_loss = 0.0
        nb_valid_examples = 0

        for i, (f_v, y_v) in enumerate(valid_dataloader):
            f_v = valid_dataset.to_device(f_v, device)
            y_v = y_v.to(device=device)

            valid_loss += loss_fn(
                ltc(f_v, valid_dataset.delta_t), y_v, "sum"
            ).item()

            if callback_batch_iter is not None:
                callback_batch_iter(i, len(valid_dataset))

            nb_valid_examples += y_v.size(0)

        ltc.train()

        return valid_loss / nb_valid_examples
