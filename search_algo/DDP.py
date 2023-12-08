import torch
import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
from torch_geometric.loader import DataLoader
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import os
from copy import deepcopy
from settings.config_file import Batch_Size
from torch.utils.data import Dataset
from settings.config_file import *


def prepare_dataloader(dataset: Dataset, batch_size: int):
    set_seed()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )


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
        self.type_model=type_model
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
            dist.barrier()
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
        return self.model


def ddp_module(total_epochs: int, model_to_train, optimizer, train_data, criterion, model_trainer, model_tester,type_model):
    set_seed()
    train_dataloader = prepare_dataloader(train_data, Batch_Size)
    trainer = Trainer(model_to_train, train_dataloader, optimizer, criterion, model_trainer, model_tester,type_model)
    model = trainer.train(total_epochs)
    return model
