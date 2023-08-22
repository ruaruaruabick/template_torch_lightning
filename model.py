import torch
import lightning.pytorch as pl
from torch_ema import ExponentialMovingAverage

class MyModel(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.save_hyperparameters()
        self.ema = ExponentialMovingAverage(self.diffusion.parameters(), decay=params.modelargs["ema"])
    
    def training_step(self, batch, batch_idx):
        return losses
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters())

    def on_before_zero_grad(self, *args, **kwargs):
        self.ema.to(self.device)
        self.ema.update()
    
    #optional
    def forward(self):
        return
    
