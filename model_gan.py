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
        loss_gen_all

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

    def configure_optimizers(self):
        # warm_up_with_cosine_lr
        MAX_STEP = 2000000
        WARMUP_STEP = 1000
        optimizer_g = torch.optim.AdamW(self.generator.parameters(),lr=self.optimizerargs['lr'],betas=self.optimizerargs['betas'],weight_decay=self.optimizerargs['weight_decay'])
        optimizer_d = torch.optim.AdamW(self.discriminator.parameters()),lr=self.optimizerargs['lr'],betas=self.optimizerargs['betas'],weight_decay=self.optimizerargs['weight_decay'])
        fun_sch = lambda epoch:(epoch+1) / WARMUP_STEP if epoch < WARMUP_STEP else 0.5 * ( math.cos((epoch - WARMUP_STEP) /(MAX_STEP - WARMUP_STEP) * math.pi) + 1)
        scheduler_g = torch.optim.lr_scheduler.LambdaLR(optimizer_g, fun_sch)
        scheduler_d = torch.optim.lr_scheduler.LambdaLR(optimizer_d, fun_sch)
        return [optimizer_g,optimizer_d],[scheduler_g,scheduler_d],
    
    #optional
    def forward(self):
        return
    
