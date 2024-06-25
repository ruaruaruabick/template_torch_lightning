import sys,os
sys.path.append(os.getcwd())
import itertools
import math

import torch
import lightning.pytorch as pl
from lightning.pytorch.utilities import grad_norm
from torch_ema import ExponentialMovingAverage
from deepspeed.ops.adam import FusedAdam

#GAN:https://lightning.ai/docs/pytorch/stable/model/manual_optimization.html
class MyModel(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.save_hyperparameters()
        self.config = params
        
        #Generator
        self.generator = Generator(**generatorargs)

        #Discriminator
        self.discriminator = Discriminator(**discriminatorargs)
        if torch.__version__ >= '2.0.0':
            self.discriminator = torch.compile(self.discriminator)
        
        #Loss
        self.loss = Loss()
        
        #EMA
        self.ema = ExponentialMovingAverage(self.generator.parameters(), decay=params["ema"])

        # Important: This property activates manual optimization.
        self.automatic_optimization = False
    
    def training_step(self, batch, batch_idx):
        g_opt, d_opt = self.optimizers()

        output = self.generator(batch)

        #optimize discriminator
        output_g = output.detach()
        loss_disc_all
        self.log("training/D_loss",loss_disc_all,prog_bar = True)
        
        d_opt.zero_grad()
        self.manual_backward(loss_disc_all)
        # self.clip_gradients(d_opt,0.,'value')
        self.log_gradients(self.discriminator,"d")
        d_opt.step()

        #optimize generator
        loss_gen_all = loss
        self.log("training/G_loss",loss_gen_all,prog_bar = True,)
        
        g_opt.zero_grad()
        self.ema.to(self.device)
        self.ema.update()
        self.manual_backward(loss_gen_all)
        # self.clip_gradients(d_opt,0.,'value')
        self.log_gradients(self.generator,"g")
        g_opt.step()

        self.log("loss",losses['loss'],prog_bar = True,)
        self.log_dict({"global_step":self.trainer.global_step},prog_bar = True)
        return losses
    
    def validation_step(self, batch, batch_idx):
        with self.ema.average_parameters():
            #forward
            
        self.log("loss",losses['loss'],sync_dist=True,prog_bar = True)
        return losses['loss']
    
    def forward(self, x):
        return self.generator(x)
    
    #some hooks
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
        optimizer_g = FusedAdam((self.generator.parameters()),**self.config["optimizer"])
        optimizer_d = FusedAdam((self.discriminator.parameters()),**self.config["optimizer"])
        scheduler_g = torch.optim.lr_scheduler.LambdaLR(optimizer_g, self.__cosine_scheduler)
        scheduler_d = torch.optim.lr_scheduler.LambdaLR(optimizer_d, self.__cosine_scheduler)
        return [optimizer_g,optimizer_d],[scheduler_g,scheduler_d],
    
    def on_save_checkpoint(self, checkpoint):
        checkpoint["ema"] = self.ema.state_dict()
        return checkpoint
    
    def on_train_batch_end(self, out, batch, batch_idx):
        g_sch, d_sch = self.lr_schedulers()
        g_sch.step()
        d_sch.step()
        
    def log_gradients(self, module, name, norm_type=2.0):
        norms = grad_norm(module, norm_type)
        self.log(f"gradients/{name}_norm_{norm_type}",norms[f"grad_{norm_type}_norm_total"],)
    
