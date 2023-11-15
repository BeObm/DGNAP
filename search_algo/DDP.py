import torch
import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
from torch_geometric.loader import DataLoader
import torch
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
from settings.config_file import Batch_Size
from torch.utils.data import Dataset
from settings.config_file import *

def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

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
            train_model
    ) -> None:
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model = model.to(self.gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_model = train_model

        self.model = DDP(self.model, device_ids=[self.gpu_id], find_unused_parameters=True)
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"total number of parameter={pytorch_total_params}")

    def _run_batch(self, data):
        self.optimizer.zero_grad()
        output = self.model(data)
        data.y = data.y.type(torch.LongTensor).to(self.gpu_id)
        # print(f"shapes are  {data.y.shape}  and {output.shape}")
        loss = self.criterion(output, data.y)
        loss.backward()
        self.optimizer.step()
        return loss

    def _run_epoch(self, epoch):
        # print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {Batch_Size} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        loss = self.train_model(model=self.model,
                         data=self.train_data,
                         criterion=self.criterion,
                         optimizer=self.optimizer,
                         devise=self.gpu_id)

        print(f"Loss value at epoch {epoch + 1}: {loss}")
    def train(self, total_epochs: int):
        for epoch in range(total_epochs):
            self._run_epoch(epoch)



def ddp_module(total_epochs: int, model_to_train, optimizer, train_data, criterion, model_trainer):

    train_dataloader = prepare_dataloader(train_data, Batch_Size)
    set_seed()
    trainer = Trainer(model_to_train, train_dataloader, optimizer, criterion, model_trainer)
    trainer.train(total_epochs)
    # destroy_process_group()
    return trainer.model
