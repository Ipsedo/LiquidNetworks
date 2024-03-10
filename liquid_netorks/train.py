# -*- coding: utf-8 -*-
import mlflow
import torch as th
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import HouseholdPowerDataset
from .metrics import Metric
from .options import ModelOptions, TrainOptions


def train(model_options: ModelOptions, train_options: TrainOptions) -> None:
    # pylint: disable=too-many-locals

    with mlflow.start_run(run_name=train_options.run_name):

        mlflow.log_params(
            {
                **model_options.to_dict(),
                **train_options.to_dict(),
            }
        )

        dataset = HouseholdPowerDataset(train_options.csv_path)
        dataloader = DataLoader(
            dataset,
            batch_size=train_options.batch_size,
            shuffle=True,
            num_workers=6,
            drop_last=True,
        )

        ltc = model_options.get_model()
        optim = th.optim.SGD(ltc.parameters(), lr=train_options.learning_rate)

        if model_options.cuda:
            device = th.device("cuda")
            th.backends.cudnn.benchmark = True
        else:
            device = th.device("cpu")

        ltc.to(device=device)

        loss_metric = Metric(train_options.metric_window_size)

        tqdm_bar = tqdm(
            range(
                train_options.epoch * len(dataset) // train_options.batch_size
            )
        )

        for e in range(train_options.epoch):

            for f, t, y in dataloader:

                f = f.to(device=device)
                t = t.to(device=device)
                y = y.to(device=device)

                out = ltc(f, t)
                loss = (
                    F.mse_loss(out.squeeze(1), y, reduction="none")
                    .sum(dim=-1)
                    .mean()
                )

                optim.zero_grad(set_to_none=True)
                loss.backward()
                optim.step()

                loss_metric.add_result(loss.item())
                grad_norm = ltc.grad_norm()

                mlflow.log_metrics(
                    {
                        "loss": loss_metric.get_last_metric(),
                        "loss_smoothed": loss_metric.get_smoothed_metric(),
                        "grad_norm": grad_norm,
                    },
                    step=tqdm_bar.n,
                )

                tqdm_bar.set_description(
                    f"Epoch {e:03} : "
                    f"loss = {loss_metric.get_last_metric():.4f}, "
                    f"loss_smoothed = {loss_metric.get_smoothed_metric():.4f}, "
                    f"grad_norm = {grad_norm:.4f}"
                )

                tqdm_bar.update(1)
