import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
import lightning as L
import hydra
from omegaconf import DictConfig
import rootutils
import logging
from torchvision.datasets import CIFAR10
from torchvision import transforms as tf
import torchmetrics as tm
logging.basicConfig(level=logging.INFO)

# import from src after this line
root_path = rootutils.setup_root(
    __file__, indicator=".project_root", pythonpath=True)
config_path = str(root_path / "config")


class CIFAR10DataModule(L.LightningDataModule):
    def __init__(
        self,
        train_transform: tf.Compose = None,
        test_transform: tf.Compose = None,
        data_dir: str = "./data",
        batch_size: int = 32,
        num_workers: int = 8
    ):
        super().__init__()
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        # download
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # dataset
        dataset = CIFAR10(
            self.data_dir, train=True, download=True,
            transform=self.train_transform)
        logging.info(f"dataset: {len(dataset)}")
        self.train_ds, self.val_ds = random_split(
            dataset, [45000, 5000])
        self.test_ds = CIFAR10(
            self.data_dir, train=False,
            download=True, transform=self.test_transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, batch_size=self.batch_size,
            num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, batch_size=self.batch_size,
            num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(
            self.test_ds, batch_size=self.batch_size,
            num_workers=self.num_workers)


# define the LightningModule
# update 1: use resnet18 as backbone, use augmentation --> 68% acc on test
# update 2: transfer learning & fine tuning --> 78% acc on test
# update 3: upsample to 224x224 before passing to resnet18 --> 86% acc on test
# update 4: upgrade to resnet50 --> 94% acc on test
class TransferLearningModule(L.LightningModule):
    def __init__(
        self,
        backbone: nn.Module,
        classification_head: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        num_classes: int = 10,
        n_transfer_epochs: int = 5
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["backbone", "classification_head"])
        self.feature_extractor, self.fc = \
            self.add_classification_head(
                backbone, classification_head, num_classes)
        self.is_finetuning = False

        self.train_loss = tm.MeanMetric()
        self.val_loss = tm.MeanMetric()
        self.test_loss = tm.MeanMetric()

        self.train_acc = tm.Accuracy(
            task="multiclass", num_classes=num_classes)
        self.val_acc = tm.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = tm.Accuracy(task="multiclass", num_classes=num_classes)

        self.val_acc_best = tm.MaxMetric()

    def add_classification_head(
        self, backbone: nn.Module, fc: nn.Module, num_classes: int
    ):
        logging.info("adding classification to backbone")
        layers = list(backbone.children())[:-1]
        feature_extractor = nn.Sequential(*layers)
        if fc is None:
            fc = nn.Sequential(
                nn.BatchNorm1d(backbone.fc.in_features),
                nn.Linear(backbone.fc.in_features, num_classes)
            )
        feature_extractor.requires_grad_(False)
        feature_extractor.eval()
        return feature_extractor, fc

    def forward(self, x):
        # upsample from 32x32 to 224x224
        x = nn.Upsample(size=(224, 224), mode="bilinear")(x)
        if self.is_finetuning:
            z = self.feature_extractor(x)
        else:
            with torch.no_grad():
                z = self.feature_extractor(x)

        z = z.view(z.size(0), -1)
        out = self.fc(z)
        return out

    def on_train_start(self):
        self.val_acc.reset()
        self.val_acc_best.reset()

    def step(self, batch, loss_metric=None, acc_metric=None):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        if loss_metric is not None:
            loss_metric(loss)
        if acc_metric is not None:
            acc_metric(preds, y)
        return loss, preds, y

    def training_step(self, batch):
        # training_step defines the train loop.
        # it is independent of forward
        loss, _, _ = self.step(batch, self.train_loss, self.train_acc)
        # Logging to TensorBoard (if installed) by default
        self.log("train/loss", self.train_loss, on_step=False,
                 on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False,
                 on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_start(self):
        if self.current_epoch >= self.hparams.n_transfer_epochs:
            if not self.is_finetuning:
                logging.info("unfreezing the feature extractor")
                self.feature_extractor.requires_grad_(True)
                self.feature_extractor.train()
                self.is_finetuning = True
        else:
            self.is_finetuning = False

    def validation_step(self, batch):
        self.step(batch, self.val_loss, self.val_acc)
        self.log("val/loss", self.val_loss, on_step=False,
                 on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False,
                 on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        acc = self.val_acc.compute()
        self.val_acc_best(acc)
        self.log("val/acc_best", self.val_acc_best, on_step=False,
                 on_epoch=True, prog_bar=True)

    def test_step(self, batch):
        self.step(batch, self.test_loss, self.test_acc)
        self.log("test/loss", self.test_loss, on_step=False,
                 on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False,
                 on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return optimizer


@hydra.main(version_base=None,
            config_path=config_path,
            config_name="basic_lvl3")
def main(cfg: DictConfig) -> None:
    logging.info("loaded config")

    model: TransferLearningModule = hydra.utils.instantiate(cfg.model)
    data_module: CIFAR10DataModule = hydra.utils.instantiate(cfg.data_module)
    trainer: L.Trainer = hydra.utils.instantiate(cfg.trainer)

    trainer.fit(model=model, datamodule=data_module)
    trainer.test(model=model, datamodule=data_module)


if __name__ == "__main__":
    main()
