import numpy as np
import torch
TrainerArgs = {
    # see https://lightning.ai/docs/pytorch/stable/common/trainer.html

    #config devices
    'accelerator' : 'gpu', # or'gpu'
    "precision" : "16", # half
    "devices" : [],
    "strategy":"ddp_find_unused_parameters_true",

    # for debugging
    "fast_dev_run" : False, #runs only 1 training and 1 validation batch and the program ends
    "limit_train_batches":.1, #

    "log_every_n_steps" : 50,
    "max_steps" : 10000, # or set max_epochs
    "accumulate_grad_batches" : 1, #total_batch = batch * accumulate_grad_batches * num_batch

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
class Config(object):
    seed = 42
    modelargs = ModelArgs
    dataloaderargs = dataloaderArgs
    trainerargs = TrainerArgs
    pass