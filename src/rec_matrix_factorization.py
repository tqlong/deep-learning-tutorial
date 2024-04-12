import rootutils
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader, Dataset
import lightning as L
import urllib.request as request
from pathlib import Path
import os
from zipfile import ZipFile
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import torchmetrics as tm


logging.basicConfig(level=logging.INFO)
# import from src after this line
root_path = rootutils.setup_root(
    __file__, indicator=".project_root", pythonpath=True)
config_path = str(root_path / "config")


class MovieLensDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        values = self.df.iloc[idx].values
        return dict(
            user=int(values[0]) - 1, item=int(values[1]) - 1, rating=np.float32(values[2] / 5.0)
        )


class MovieLensDataModule(L.LightningDataModule):
    def __init__(
        self,
        root_path,
        data_relative_path,
        data_url="https://files.grouplens.org/datasets/movielens/ml-100k.zip",
        batch_size=64,
        num_workers=4,
        random_state=42
    ):
        super().__init__()
        self.save_hyperparameters()

    def prepare_data(self):
        # download zip data from data_url to root_path
        root_path = Path(self.hparams.root_path)
        zip_file_name = os.path.basename(self.hparams.data_url)
        zip_file_path = root_path / zip_file_name
        if Path.exists(zip_file_path):
            logging.info(f"File {zip_file_path} already exists")
        else:
            logging.info(f"Downloading {self.hparams.data_url} to {zip_file_path}")
            request.urlretrieve(self.hparams.data_url, zip_file_path, reporthook=self.download_hook)
            logging.info("Data downloaded")

        # unzip the file
        if Path.exists(zip_file_path) and not Path.exists(root_path / "ml-100k"):
            with ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(root_path)
            logging.info("Data unzipped")
        else:
            logging.info(f"File {zip_file_path} already unzipped")

    def download_hook(self, block_num, block_size, total_size):
        logging.info(f"Downloading {block_num * block_size / total_size * 100:.2f}%")

    def setup(self, stage: str):
        # read into dataframe
        root_path = Path(self.hparams.root_path)
        data_path = root_path / self.hparams.data_relative_path
        df = pd.read_csv(data_path, sep="\t", header=None)

        # split into train, val, test
        train_df, test_df = train_test_split(df, test_size=0.1, random_state=self.hparams.random_state)
        train_df, val_df = train_test_split(train_df, test_size=1 / 9, random_state=self.hparams.random_state)

        # create datasets
        if stage == "fit" or stage is None:
            self.train_dataset = MovieLensDataset(train_df)
            self.val_dataset = MovieLensDataset(val_df)
        if stage == "test" or stage is None:
            self.test_dataset = MovieLensDataset(test_df)

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
    def __init__(self, num_users, num_items, optimizer, n_hidden=100, dropout=0.5, embedding_dim=32):
        super().__init__()
        self.save_hyperparameters()
        self.user_embeddings = torch.nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = torch.nn.Embedding(num_items, embedding_dim)
        self.score_net = torch.nn.Sequential(
            # torch.nn.BatchNorm1d(2 * embedding_dim),
            torch.nn.Linear(2 * embedding_dim, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            # torch.nn.Linear(n_hidden, n_hidden),
            # torch.nn.ReLU(),
            # torch.nn.Dropout(dropout),
            torch.nn.Linear(n_hidden, 1)
        )
        self.initialize_parameters()

        self.train_loss = tm.MeanMetric()
        self.val_loss = tm.MeanMetric()
        self.test_loss = tm.MeanMetric()

        self.train_mae = tm.MeanAbsoluteError()
        self.val_mae = tm.MeanAbsoluteError()
        self.val_mae_best = tm.MinMetric()
        self.test_mae = tm.MeanAbsoluteError()

    def init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    def initialize_parameters(self):
        torch.nn.init.normal_(self.user_embeddings.weight, std=0.01)
        torch.nn.init.normal_(self.item_embeddings.weight, std=0.01)
        self.score_net.apply(self.init_weights)

    def forward(self, user, item):
        user_embedding = self.user_embeddings(user)
        item_embedding = self.item_embeddings(item)
        # x_term = user_embedding * item_embedding
        x = torch.cat([user_embedding, item_embedding], dim=1)
        out = self.score_net(x)
        return out.squeeze(1)

    def step(self, batch, loss_metric, mae_metric):
        user, item, rating = batch['user'], batch['item'], batch['rating']
        prediction = self(user, item)
        loss = torch.nn.functional.mse_loss(prediction, rating)
        loss_metric(loss)
        mae_metric(prediction, rating)
        return loss, prediction, rating

    def on_train_start(self) -> None:
        self.val_mae.reset()
        self.val_mae_best.reset()

    def on_validation_epoch_end(self):
        mae = self.val_mae.compute()
        self.val_mae_best(mae)
        self.log("val/mae_best", self.val_mae_best.compute(), sync_dist=True, prog_bar=True)

    def training_step(self, batch, batch_idx):
        loss, prediction, rating = self.step(batch, self.train_loss, self.train_mae)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/mae", self.train_mae, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, prediction, rating = self.step(batch, self.val_loss, self.val_mae)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/mae", self.val_mae, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        loss, prediction, rating = self.step(batch, self.test_loss, self.test_mae)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test/mae", self.test_mae, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(self.parameters())
        return optimizer


@hydra.main(version_base=None, config_path=config_path, config_name="rec")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
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
