"""Matrix Factorization model for recommendation

This script trains a matrix factorization model for recommendation using the MovieLens dataset.
The model is trained using PyTorch Lightning.
The configuration is done using Hydra (data module, model, callbacks, trainer).
The data module downloads the MovieLens dataset, splits it into train, val, test sets.
The model is a matrix factorization model with a score network.
The model is trained and tested using MSE loss and evaluated using MAE metric.
The trainer is configured to use a GPU if available and to use a checkpoint callback to save the best model.
"""
import logging
import os
from pathlib import Path
from zipfile import ZipFile

import hydra
import lightning as L
import numpy as np
import pandas as pd
import rootutils
import torch
import torchmetrics as tm
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from urllib import request


logging.basicConfig(level=logging.INFO)
# Set up the root path of the project.
# import from src after this line
ROOT_PATH = rootutils.setup_root(
    __file__, indicator=".project_root", pythonpath=True)
CONFIG_PATH = str(ROOT_PATH / "config")


class DownloadHook:
    """Wraps tqdm instance.
    Adapt from: https://gist.github.com/leimao/37ff6e990b3226c2c9670a2cd1e4a6f5
    Don't forget to close() or __exit__()
    the tqdm instance once you're done with it (easiest using `with` syntax).
    Example
    -------
    >>> with tqdm(...) as t:
    ...     reporthook = my_hook(t)
    ...     urllib.urlretrieve(..., reporthook=reporthook)
    """

    def __init__(self, t: tqdm):
        self.t = t
        self.last_b = 0

    def __call__(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.t.total = tsize
        self.t.update((b - self.last_b) * bsize)
        self.last_b = b


class MovieLensDataset(Dataset):
    """MovieLens dataset
    """

    def __init__(self, df):
        """Initialize the dataset

        Args:
            df (pd.DataFrame): dataframe with columns user, item, rating, timestamp
        """
        self.df = df

    def __len__(self):
        """Return the length of the dataset

        Returns:
            int: length of the dataset
        """
        return len(self.df)

    def __getitem__(self, idx):
        """Get the item at index idx

        Args:
            idx (int): index

        Returns:
            dict: dictionary with keys user, item, rating

        Note:
            user and item are 0-indexed
            rating is normalized to [0, 1]
        """
        values = self.df.iloc[idx].values
        return dict(
            user=int(values[0]) - 1, item=int(values[1]) - 1, rating=np.float32(values[2] / 5.0)
        )


class MovieLensDataModule(L.LightningDataModule):
    """MovieLens lightning data module"""

    def __init__(
        self,
        root_path,
        data_relative_path,
        data_url,
        batch_size=64,
        num_workers=4,
    ):
        """Initialize the data module

        Args:
            root_path (_type_): the root path of the data (containing the zip file)
            data_relative_path (_type_): the relative path of the data file within the root path
            data_url (str, optional): url to download.
            batch_size (int, optional): batch size for dataloaders. Defaults to 64.
            num_workers (int, optional): number of workers for dataloaders. Defaults to 4.
        """
        super().__init__()
        self.save_hyperparameters()
        self.train_df, self.val_df, self.test_df = None, None, None
        self.train_dataset, self.val_dataset, self.test_dataset = None, None, None

    def prepare_data(self):
        """Download and unzip the data if not already downloaded and unzipped
        Zip file is downloaded from data_url and unzipped to root_path
        Check if the data file already exists in the data relative path an unzip if not
        """
        # download zip data from data_url to root_path
        root_path = Path(self.hparams.root_path)
        zip_file_name = os.path.basename(self.hparams.data_url)
        zip_file_path = root_path / zip_file_name
        if Path.exists(zip_file_path):
            logging.info(f"File {zip_file_path} already exists")
        else:
            logging.info(f"Downloading {self.hparams.data_url} to {zip_file_path}")

            with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=str(zip_file_path)) as t:
                request.urlretrieve(self.hparams.data_url, zip_file_path, reporthook=DownloadHook(t), data=None)
            logging.info("Data downloaded")

        # unzip the file
        data_root = root_path / self.hparams.data_relative_path.split("/")[0]
        if Path.exists(zip_file_path) and not Path.exists(data_root):
            with ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(root_path)
            logging.info("Data unzipped")
        else:
            logging.info(f"File {zip_file_path} already unzipped")

    def setup(self, stage: str):
        """Setup the data module
        Split the data frame into train, val, test
        Create the train, val, test datasets
        """
        # read into dataframe
        root_path = Path(self.hparams.root_path)
        data_path = root_path / self.hparams.data_relative_path
        df = pd.read_csv(data_path, sep="\t", header=None)
        print(df.max(axis=0), df.shape)

        # split into train, val, test
        if self.train_df is None or self.val_df is None or self.test_df is None:
            train_df, self.test_df = train_test_split(df, test_size=0.1)
            self.train_df, self.val_df = train_test_split(train_df, test_size=1 / 9)

        # create datasets
        if stage == "fit" or stage is None:
            self.train_dataset = MovieLensDataset(self.train_df)
            self.val_dataset = MovieLensDataset(self.val_df)
        if stage == "test" or stage is None:
            self.test_dataset = MovieLensDataset(self.test_df)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size,
                          shuffle=True, num_workers=self.hparams.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size,
                          shuffle=False, num_workers=self.hparams.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size,
                          shuffle=False, num_workers=self.hparams.num_workers)


class MatrixFactorization(L.LightningModule):
    """Matrix Factorization model"""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        score_net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        n_hidden=100,
        embedding_dim=32
    ):
        """Initialize the model

        Args:
            num_users (int): number of users
            num_items (int): number of items
            score_net (torch.nn.Module): score network
            optimizer: optimizer
            n_hidden (int, optional): number of hidden units. Defaults to 100.
            embedding_dim (int, optional): embedding dimension. Defaults to 32.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["score_net"])
        self.user_embeddings = torch.nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = torch.nn.Embedding(num_items, embedding_dim)
        self.score_net = score_net
        self.initialize_parameters()

        self.train_loss = tm.MeanMetric()
        self.val_loss = tm.MeanMetric()
        self.test_loss = tm.MeanMetric()

        self.train_mae = tm.MeanAbsoluteError()
        self.val_mae = tm.MeanAbsoluteError()
        self.val_mae_best = tm.MinMetric()
        self.test_mae = tm.MeanAbsoluteError()

    def init_weights(self, m):
        """Initialize the weights of the score model layer"""
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    def initialize_parameters(self):
        """Initialize the parameters of the model"""
        torch.nn.init.normal_(self.user_embeddings.weight, std=0.01)
        torch.nn.init.normal_(self.item_embeddings.weight, std=0.01)
        self.score_net.apply(self.init_weights)

    def forward(self, user, item):
        """Forward pass

        Args:
            user (torch.Tensor): user index, size [B]
            item (torch.Tensor): item index, size [B]

        Returns:
            torch.Tensor: score prediction, size [B]

        Note:
            B is the batch size
            user and item are 0-indexed
        """
        user_embedding = self.user_embeddings(user)
        item_embedding = self.item_embeddings(item)
        # x_term = user_embedding * item_embedding
        x = torch.cat([user_embedding, item_embedding], dim=1)
        out = self.score_net(x)
        return out.squeeze(1)

    def step(self, batch, loss_metric, mae_metric):
        """Step function for training, validation and testing

        Args:
            batch (dict): batch with keys user, item, rating
            loss_metric (torchmetrics.Metric): loss metric
            mae_metric (torchmetrics.Metric): mae metric

        Returns:
            tuple: loss, prediction, target rating
        """
        user, item, rating = batch['user'], batch['item'], batch['rating']
        prediction = self(user, item)
        loss = torch.nn.functional.mse_loss(prediction, rating)
        loss_metric(loss)
        mae_metric(prediction, rating)
        return loss, prediction, rating

    def on_train_start(self):
        """Reset the validation loss and mae metrics in case the test run initializes them"""
        self.val_mae.reset()
        self.val_mae_best.reset()

    def on_validation_epoch_end(self):
        """Update the best validation mae metric after each validation epoch"""
        mae = self.val_mae.compute()
        self.val_mae_best(mae)
        self.log("val/mae_best", self.val_mae_best.compute(),
                 sync_dist=True, prog_bar=True)

    def training_step(self, batch, batch_idx):
        """Training step

        Args:
            batch (dict): batch with keys user, item, rating
            batch_idx (int): batch index

        Returns:
            loss (torch.Tensor): mse loss

        Note:
            The training loss and mae are computed and logged
        """
        loss, prediction, rating = self.step(batch, self.train_loss, self.train_mae)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/mae", self.train_mae, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step

        Args:
            batch (dict): batch with keys user, item, rating
            batch_idx (int): batch index

        Note:
            The validation loss and mae are computed and logged
        """
        loss, prediction, rating = self.step(batch, self.val_loss, self.val_mae)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/mae", self.val_mae, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        """Test step

        Args:
            batch (dict): batch with keys user, item, rating
            batch_idx (int): batch index

        Note:
            The test loss and mae are computed and logged
        """
        loss, prediction, rating = self.step(
            batch, self.test_loss, self.test_mae)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test/mae", self.test_mae, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        """Configure the optimizer

        Note:
            The optimizer is patially configured from the hparams
            At this point, the optimizer needs to be fully configured
            using the parameters of the model
        """
        optimizer = self.hparams.optimizer(self.parameters())
        return optimizer


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="rec")
def main(cfg: DictConfig) -> None:
    """Main function to train and test the model

    Args:
        cfg (DictConfig): configuration object from config file
        cfg.data_module (DictConfig): data module configuration
        cfg.model (DictConfig): model configuration
        cfg.callbacks (List[DictConfig]): list of callback configurations
        cfg.trainer (DictConfig): trainer configuration
    """
    print(OmegaConf.to_yaml(cfg))
    L.seed_everything(cfg.seed)

    data_module: L.LightningDataModule = hydra.utils.instantiate(cfg.data_module)
    model: L.LightningModule = hydra.utils.instantiate(cfg.model)
    callbacks = [hydra.utils.instantiate(cb) for cb in cfg.callbacks]
    trainer: L.Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks)

    trainer.fit(model, data_module)

    ckpt_path = trainer.checkpoint_callback.best_model_path
    if ckpt_path == "":
        logging.info("no best model found, using current model")
        ckpt_path = None
    trainer.test(model=model, datamodule=data_module, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
