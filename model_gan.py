import math
import torch
import lightning.pytorch as pl
from torch_ema import ExponentialMovingAverage
#GAN:https://lightning.ai/docs/pytorch/2.0.0/notebooks/lightning_examples/basic-gan.html
class MyModel(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.save_hyperparameters()

        #Generator
        self.generator = Generator(**generatorargs)

        #Discriminator
        self.discriminator = Discriminator(**discriminatorargs)

        #EMA
        self.ema_g = ExponentialMovingAverage(self.model.parameters(), decay=params.modelargs["ema"])
        self.ema_d = ExponentialMovingAverage(self.model.parameters(), decay=params.modelargs["ema"])

        # Important: This property activates manual optimization.
        self.automatic_optimization = False
    
    def training_step(self, batch, batch_idx):
        g_opt, d_opt = self.optimizers()

        output = self.generator(batch)

        #optimize discriminator
        loss_disc_all

        d_opt.zero_grad()
        self.ema_d.to(self.device)
        self.ema_d.update()
        self.manual_backward(loss_disc_all)
        #gradient clip 
        self.clip_gradients(d_opt,1.0,'norm')
        d_opt.step()

        #optimize generator
        loss_gen_all = loss

        g_opt.zero_grad()
        self.ema_g.to(self.device)
        self.ema_g.update()
        self.manual_backward(loss_gen_all)
        #gradient clip 
        self.clip_gradients(g_opt,1.0,'norm')
        # self.get_grad_dict(self.generator,'g')
        g_opt.step()

        self.log("loss",losses['loss'],prog_bar = True,)
        self.log_dict({"global_step":self.trainer.global_step},prog_bar = True)
        return losses

    def validation_step(self, batch, batch_idx):
        with self.ema_g.average_parameters():
            #forward
            
        self.log("loss",losses['loss'],sync_dist=True,prog_bar = True, rank_zero_only = True)
        return losses['loss']
    
    def test_step(self, batch, batch_idx):
        return losses['loss']
    
    def on_train_batch_end(self, out, batch, batch_idx):
        g_sch, d_sch = self.lr_schedulers()
        g_sch.step()
        d_sch.step()

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
        optimizer_g = torch.optim.AdamW(self.generator.parameters(),**self.config["optimizer"])
        optimizer_d = torch.optim.AdamW(self.discriminator.parameters(),**self.config["optimizer"])
        scheduler_g = torch.optim.lr_scheduler.LambdaLR(optimizer_g, self.__cosine_scheduler)
        scheduler_d = torch.optim.lr_scheduler.LambdaLR(optimizer_d, self.__cosine_scheduler)
        return [optimizer_g,optimizer_d],[scheduler_g,scheduler_d],
    
    #optional
    def forward(self):
        return
    
