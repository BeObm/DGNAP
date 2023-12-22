from collections import OrderedDict

import torch
import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
from torch_geometric.loader import DataLoader
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import re
from copy import deepcopy
from settings.config_file import Batch_Size
from torch.utils.data import Dataset
from settings.config_file import *
import torch.multiprocessing as mp
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs



def prepare_data_loader(dataset, batch_size=32,shuffle=False):
    dataloader = DataLoader(dataset, batch_size=batch_size,drop_last=False, shuffle=shuffle)
    return dataloader

    
def ddp_module(accelerator, total_epochs: int, model_to_train, optimizer, train_dataloader,criterion, model_trainer,save_path="models.pt"):
    set_seed()

    train_dataloader, model,optimizer = accelerator.prepare(train_dataloader,model_to_train,optimizer)
    model.train()
    for epoch in range(total_epochs):
        model_trainer(model=model,
                      dataloader=train_dataloader,
                      criterion=criterion,
                      optimizer=optimizer,
                      accelerator=accelerator)

    accelerator.wait_for_everyone()

    return model
