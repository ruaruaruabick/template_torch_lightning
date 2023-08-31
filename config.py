import numpy as np
import torch
import math
TrainerArgs = {
    # see https://lightning.ai/docs/pytorch/stable/common/trainer.html

    #config devices
    'accelerator' : 'gpu', # or'gpu'
    "precision" : "16-mixed", # half
    "devices" : [],
    "strategy":"ddp_find_unused_parameters_true",

    # for debugging
    "fast_dev_run" : False, #runs only 1 training and 1 validation batch and the program ends
    "limit_train_batches":.1, #1.0 for all batch

    "log_every_n_steps" : 50,
    "max_steps" : 10000, # or set max_epochs
    "accumulate_grad_batches" : 2, 
    "val_check_interval":1000,
    #total_batch = batch * accumulate_grad_batches * num_batch

    #default
    "deterministic" : True, #set true to ensure full reproducibility
    "enable_checkpointing" : True, #saves the most recent model to a single checkpoint after each epoch
    "logger" : True #ses the default TensorBoardLogger 
}

dataloaderArgs:{
    "batch_size" : 64,
    "shuffle" : True,
    "num_workers" : 0, #0 for debug
    "pin_memory" : True,
    "drop_last" : False,
}
ModelArgs:{
    "ema":.995,
}
OptimizerArgs={
    'lr':1e-5,
    'betas':(.9,.999),
    'weight_decay':.001
}
SchedulerArgs={
    'warm_up_epochs':1000,
    #cosine 
    'function':lambda epoch: epoch / SchedulerArgs['warm_up_epochs'] if epoch <= SchedulerArgs['warm_up_epochs'] else 0.5 * ( math.cos((epoch - SchedulerArgs['warm_up_epochs']) /(TrainerArgs['max_epochs'] - SchedulerArgs['warm_up_epochs']) * math.pi) + 1)
}
ConfigArgs:{
    'seed' : 42,
    'total_batch':16,
    'modelargs' : ModelArgs,
    'dataloaderargs' : dataloaderArgs,
    'trainerargs' : TrainerArgs,
    'optimizerargs':OptimizerArgs,
    'schedulerArgs':SchedulerArgs,
}

class Config(object):
    def __init__(self) -> None:
        self.configargs = ConfigArgs
        configargs['dataloaderconfig']['batch_size'] = configargs['total_batch']//len(configargs['trainerargs']["devices"])//configargs['trainerargs']["accumulate_grad_batches"]
        pass
