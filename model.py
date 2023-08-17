import torch
import lightning.pytorch as pl
class MyModel(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.save_hyperparameters()
    
    def training_step(self, batch, batch_idx):
        return losses
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.trainer.model.parameters(), lr=0.02)

    #optional
    def forward(self):
        return
    
