import torch
from torch import optim, nn
from torch.utils.data import DataLoader
import lightning as L
import hydra
from omegaconf import DictConfig, OmegaConf
import rootutils

## import from src after this line
root_path = rootutils.setup_root(__file__, indicator=".project_root", pythonpath=True)
config_path = str(root_path / "config")


# define the LightningModule
class LitAutoEncoder(L.LightningModule):
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


@hydra.main(version_base=None, config_path=config_path, config_name="15min")
def main(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    autoencoder: LitAutoEncoder = hydra.utils.instantiate(cfg.autoencoder)
    train_loader: DataLoader = hydra.utils.instantiate(cfg.train_loader)
    trainer: L.Trainer = hydra.utils.instantiate(cfg.trainer)

    trainer.fit(model=autoencoder, train_dataloaders=train_loader)

    checkpoint = "./lightning_logs/version_0/checkpoints/epoch=0-step=100.ckpt"
    autoencoder = LitAutoEncoder.load_from_checkpoint(checkpoint, encoder=autoencoder.encoder, decoder=autoencoder.decoder)
    encoder = autoencoder.encoder
    encoder.eval()

    fake_image_batch = torch.rand(4, 28 * 28, device=autoencoder.device)
    embeddings = encoder(fake_image_batch)
    print("⚡" * 20, "\nPredictions (4 image embeddings):\n", embeddings, "\n", "⚡" * 20)


if __name__ == "__main__":
    main()
