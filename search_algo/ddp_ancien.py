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


class Trainer:
    def __init__(
            self,
            model: torch.nn.Module,
            train_data: DataLoader,
            optimizer: torch.optim.Optimizer,
            criterion: torch.nn.functional,
            train_model,
            model_tester,
            type_model
    ) -> None:
        self.best_epoch = None
        self.weight_path = 'temp_best_gnn.pth'
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model = model.to(self.gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_model = train_model
        self.model_tester = model_tester
        self.type_model = type_model
        self.best_model = None
        print(f"GPU id is {self.gpu_id}")

        self.model = DDP(self.model, device_ids=[self.gpu_id], find_unused_parameters=True)
        if self.gpu_id == 0:
            pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"total number of parameter={pytorch_total_params}")

    def train(self, total_epochs: int):
        best_model = None
        if config["param"]["best_search_metric_rule"] == 'max':
            best_performance = -99999999
        else:
            best_performance = 99999999
        for epoch in range(total_epochs):
            # self._run_epoch(epoch)
            # dist.barrier()
            self.train_data.sampler.set_epoch(epoch)
            loss = self.train_model(model=self.model,
                                    data=self.train_data,
                                    criterion=self.criterion,
                                    optimizer=self.optimizer,
                                    devise=torch.device(f'cuda:{self.gpu_id}'))

            if self.gpu_id == 0:

                perf_value = self.model_tester(model=self.model,
                                               test_loader=self.train_data,
                                               devise=torch.device('cuda:0'),
                                               type_data="val")[config["param"]["search_metric"]]
                if config["param"]["best_search_metric_rule"] == "max":
                    if perf_value > best_performance:
                        best_performance = perf_value
                        self.best_epoch = epoch + 1
                        # torch.save(self.model, self.weight_path)
                elif config["param"]["best_search_metric_rule"] == "min":
                    if perf_value < best_performance:
                        best_performance = perf_value
                        # torch.save(self.model, self.weight_path)
                        self.best_epoch = epoch + 1
                # print(f'Epoch: {epoch + 1} | Loss: {loss:.4f} | {config["param"]["search_metric"]}:{perf_value:.4f}')
            dist.barrier()
        return self.model


def setup_process_group(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def ddp_checkpoint_to_non_ddp_checkpoint(state_dict):
    model_dict = OrderedDict()
    pattern = re.compile('module.')
    for k, v in state_dict.items():
        if re.search("module", k):
            model_dict[re.sub(pattern, '', k)] = v
        else:
            model_dict = state_dict
    return model_dict


def prepare_data_loader(dataset, rank, world_size, batch_size=32, pin_memory=False, num_workers=0):
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)

    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers,
                            drop_last=False, shuffle=False, sampler=sampler)
    return dataloader


def cleanup():
    dist.destroy_process_group()


def ddp_module(rank, world_size, total_epochs: int, model_to_train, optim, lr, wd, dataset, criterion, model_trainer,
               model_tester, type_model):
    set_seed()
    # setup the process groups
    setup_process_group(rank=rank, world_size=world_size)

    # prepare the dataloader
    dataloader = prepare_data_loader(dataset=dataset, rank=rank, world_size=world_size, batch_size=Batch_Size)

    # instantiate the model(it's your own model) and move it to the right device
    model = model_to_train.to(rank)

    # wrap the model with DDP
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    best_loss = 99999
    if rank == 0:
        print(f"Start training in distributed environment with {world_size} GPUs")

    optimizer = optim(model.module.parameters(), lr=lr, weight_decay=wd)
    for epoch in range(total_epochs):
        # tell DistributedSampler which epoch this is
        dataloader.sampler.set_epoch(epoch)
        loss = model_trainer(model=model,
                             dataloader=dataloader,
                             criterion=criterion,
                             optimizer=optimizer,
                             rank=rank)

        if rank == 0:
            if loss < best_loss:
                torch.save(model, weight_path)

    cleanup()
    if rank == 0:
        return model

