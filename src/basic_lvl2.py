from torch import optim, nn
from torch.utils.data import DataLoader, random_split
import lightning as L
import hydra
from omegaconf import DictConfig
import rootutils
import logging
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor  # Normalize
logging.basicConfig(level=logging.INFO)

# import from src after this line
root_path = rootutils.setup_root(
    __file__, indicator=".project_root", pythonpath=True)
config_path = str(root_path / "config")


class MNISTDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 32,
        num_workers: int = 8
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # transforms
        transform = Compose([
            ToTensor(),
            # Normalize((0.1307,), (0.3081,))
        ])

        # dataset
        mnist_train = MNIST(
            self.data_dir, train=True, download=True, transform=transform)
        self.mnist_train, self.mnist_val = random_split(
            mnist_train, [55000, 5000])
        self.mnist_test = MNIST(
            self.data_dir, train=False, download=True, transform=transform)

    def train_dataloader(self):
        return DataLoader(
            self.mnist_train, batch_size=self.batch_size,
            num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(
            self.mnist_val, batch_size=self.batch_size,
            num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(
            self.mnist_test, batch_size=self.batch_size,
            num_workers=self.num_workers)


# define the LightningModule
class LitAutoEncoder(L.LightningModule):
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        z = self.encoder(x)
        return z

    def step(self, batch):
        x, _ = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        return loss, z

    def training_step(self, batch):
        # training_step defines the train loop.
        # it is independent of forward
        x, _ = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss, on_step=False,
                 on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch):
        x, _ = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        self.log("val_loss", loss, on_step=False,
                 on_epoch=True, prog_bar=True)

    def test_step(self, batch):
        x, _y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


@hydra.main(version_base=None,
            config_path=config_path,
            config_name="basic_lvl2")
def main(cfg: DictConfig) -> None:
    logging.info("loaded config")

    autoencoder: LitAutoEncoder = hydra.utils.instantiate(cfg.autoencoder)
    data_module: MNISTDataModule = hydra.utils.instantiate(cfg.data_module)

    trainer: L.Trainer = hydra.utils.instantiate(cfg.trainer)

    trainer.fit(
        model=autoencoder,
        datamodule=data_module
    )

    checkpoint = "./lightning_logs/version_0/checkpoints/epoch=0-step=100.ckpt"
    autoencoder = LitAutoEncoder.load_from_checkpoint(
        checkpoint_path=checkpoint,
        encoder=autoencoder.encoder,
        decoder=autoencoder.decoder
    )
    encoder = autoencoder.encoder
    encoder.eval()

    trainer.test(model=autoencoder, datamodule=data_module)


if __name__ == "__main__":
    main()
