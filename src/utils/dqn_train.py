from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

from utils import plot_metrics
from utils.config import DQNConfig

root_directory = Path(__file__).parent.parent.parent


def find_latest_version(logging_name: str):
    dir_list = [x for x in (root_directory / "logs" / logging_name).iterdir() if x.is_dir()]

    dir_number = [int(x.name.split("_")[1]) for x in dir_list]
    latest_version = max(dir_number)
    return latest_version


def train_dqn(model, config: DQNConfig):
    logging_name = "_".join(config.env.split("/"))

    trainer = pl.Trainer(
        accelerator=config.device,
        max_epochs=config.max_epochs,
        logger=CSVLogger(str(root_directory / "logs"), name=logging_name),
    )

    trainer.fit(model)

    latest_version = find_latest_version(logging_name)

    plot_metrics(str(root_directory / "logs" / logging_name / f"version_{latest_version}" / "metrics.csv"))
