from collections import OrderedDict

import torch
import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
from torch_geometric.loader import DataLoader,ClusterLoader, ClusterData, NeighborLoader
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




def prepare_data_loader(data, batch_size=Batch_Size,shuffle=False):

    if config["dataset"]["type_task"] =="node_classification":
        cluster_data = ClusterData(data, num_parts=ncluster)
        dataloader = ClusterLoader(cluster_data, batch_size=batch_size, shuffle=True)
        # loader = NeighborLoader(
        #     data,
        #     # Sample 30 neighbors for each node for 2 iterations
        #     num_neighbors=[30] * 2,
        #     # Use a batch size of 128 for sampling training nodes
        #     batch_size=128,
        #     input_nodes=None,
        # )
    else:
        dataloader = DataLoader(data, batch_size=batch_size,drop_last=False, shuffle=shuffle)
    return dataloader

    
def ddp_module(accelerator, total_epochs: int, model_to_train, optimizer, train_dataloader,criterion, model_trainer):
    set_seed()

    train_dataloader, model,optimizer = accelerator.prepare(train_dataloader,model_to_train,optimizer)
    model.train()
    for epoch in range(total_epochs):
        model_trainer(model=model,
                      dataloader=train_dataloader,
                      criterion=criterion,
                      optimizer=optimizer,
                      accelerator=accelerator)

    return model

 #
    # accelerator.wait_for_everyone()
    # unwrapped_model = accelerator.unwrap_model(model)
    # accelerator.save(unwrapped_model.state_dict(), filename)

 # base_model = (model.module if isinstance(model, DistributedDataParallel) else model)
 #        checkpoint_dir = tempfile.mkdtemp()
 #        torch.save(
 #            {"model_state_dict": base_model.state_dict()},
 #            os.path.join(checkpoint_dir, "model.pt"),
 #        )
 #        checkpoint = Checkpoint.from_directory(checkpoint_dir)