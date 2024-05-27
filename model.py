import math
import torch
import lightning.pytorch as pl
from torch_ema import ExponentialMovingAverage
from lightning.pytorch.utilities import grad_norm

class MyModel(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.save_hyperparameters()
        self.config = params
        self.ema = ExponentialMovingAverage(self.model.parameters(), decay=params.modelargs["ema"])
    
    def training_step(self, batch, batch_idx):
        self.log("loss",losses['loss'],prog_bar = True,)
        self.log_dict({"global_step":self.trainer.global_step},prog_bar = True)
        return losses

    def validation_step(self, batch, batch_idx):
        with self.ema.average_parameters():
            #forward
        self.log("loss",losses['loss'],sync_dist=True,prog_bar = True)

        logger = self.logger.experiment
        step = self.trainer.global_step

        logger.add_figure("name", fig, step)
        logger.add_audio("name",wav,step,sample_rate=,)
        
        plt.close()
        return losses['loss']
    
    def test_step(self, batch, batch_idx):
        return losses['loss']
    
    def __cosine_scheduler(self,step):
        init_lr = self.config["optimizer"]["lr"]
        last_lr = self.config["scheduler"]["last_lr"]
        warmupstep = self.config["scheduler"]["warmup_steps"]
        totalstep = self.config["trainer"]["max_steps"]
        if step < warmupstep:
            return (step+1) / warmupstep
        else:
            cosine_rate = 0.5 * ( math.cos((step - warmupstep) /(totalstep - warmupstep) * math.pi) + 1)
            return (cosine_rate * (init_lr - last_lr)+last_lr) / init_lr
        
    def configure_optimizers(self):
        # warm_up_with_cosine_lr
        optimizer = torch.optim.AdamW(self.parameters(),**self.config["optimizer"])
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, self.__cosine_scheduler)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                'interval': 'step', # or 'epoch'
                'frequency': 1,
            },
        }

    def on_before_zero_grad(self, *args, **kwargs):
        self.ema.to(self.device)
        self.ema.update()
    
    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        norms = grad_norm(self.layer, norm_type=2)
        
        def get_norm(self, model, threshold=1.0):
            norms = grad_norm(model, norm_type=2)
            norms = {k:v for k,v in norms.items() if v.item() > threshold}
            return norms
        
        norms = get_norm(self.model)
        self.log_dict(norms)
    
    #optional
    def forward(self):
        return
    
