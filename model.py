import torch
import lightning.pytorch as pl
from torch_ema import ExponentialMovingAverage
#GAN:https://lightning.ai/docs/pytorch/2.0.0/notebooks/lightning_examples/basic-gan.html
class MyModel(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.save_hyperparameters()
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
    
    def configure_optimizers(self):
        # warm_up_with_cosine_lr
        optimizer = torch.optim.AdamW(self.model.parameters(),lr=self.args.optimizerargs['lr'],betas=self.args.optimizerargs['betas'],weight_decay=self.args.optimizerargs['weight_decay'])
        scheduler = torch.optim.lr_scheduler.LambdaLR(lambda epoch: epoch / SchedulerArgs['warm_up_epochs'] if epoch <= SchedulerArgs['warm_up_epochs'] 
                                                      else 0.5 * ( math.cos((epoch - SchedulerArgs['warm_up_epochs']) /(TrainerArgs['max_epochs'] - SchedulerArgs['warm_up_epochs']) * math.pi) + 1), self.args.schedulerArgs['function'])
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
    
    #optional
    def forward(self):
        return
    
