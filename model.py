import torch
import lightning.pytorch as pl
from torch_ema import ExponentialMovingAverage

class MyModel(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.save_hyperparameters()
        self.ema = ExponentialMovingAverage(self.diffusion.parameters(), decay=params.modelargs["ema"])
    
    def training_step(self, batch, batch_idx):

        self.log("loss",losses['loss'])
        if self.trainer.global_step % 50 == 0:
            print("step:{},loss:{}".format(self.trainer.global_step,losses['loss']))

        return losses
    
    def validation_step(self, batch, batch_idx):
        self.diffusion.eval()
            with torch.no_grad():
        self.diffusion.train()
        return losses['loss']
    
    def configure_optimizers(self):
        # warm_up_with_cosine_lr
        optimizer = torch.optim.AdamW(self.diffusion.parameters(),lr=self.args.optimizerargs['lr'],betas=self.args.optimizerargs['betas'],weight_decay=self.args.optimizerargs['weight_decay'])
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, self.args.schedulerArgs['function'])
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
            },
        }

    def on_before_zero_grad(self, *args, **kwargs):
        self.ema.to(self.device)
        self.ema.update()
    
    #optional
    def forward(self):
        return
    
