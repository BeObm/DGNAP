import torch
import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
from torch_geometric.loader import DataLoader
import torch
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
from settings.config_file import Batch_Size
from torch.utils.data import Dataset



def ddp_setup():

    init_process_group(backend="gloo")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))



class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        save_every: int,
        snapshot_path: str,
        criterion:torch.nn.functional,
        train_batch
    ) -> None:
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model = model.to(self.gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.epochs_run = 0
        self.snapshot_path = snapshot_path
        self.criterion=criterion
        self.train_batch=train_batch
        if os.path.exists(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(snapshot_path)

        self.model = DDP(self.model, device_ids=[self.gpu_id],find_unused_parameters=True)
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"total number of parameter={pytorch_total_params}")
    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")


    def _run_batch(self, data):

        self.optimizer.zero_grad()
        output = self.model(data)
        data.y = data.y.type(torch.LongTensor).to(self.gpu_id)
        # print(f"shapes are  {data.y.shape}  and {output.shape}")
        loss =  self.criterion(output, data.y)
        loss.backward()
        self.optimizer.step()
        return loss

    def _run_epoch(self, epoch):

        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {Batch_Size} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        for data in self.train_data:
            data = data.to(self.gpu_id)
            # source = source.to(self.gpu_id)
            # targets = targets.to(self.gpu_id)
            self.train_batch(model=self.model,
                                    data=data,
                                    criterion=self.criterion,
                                    optimizer=self.optimizer)





    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)


def load_train_objs():
    train_set = MyTrainDataset(2048)  # load your dataset
    model = torch.nn.Linear(20, 1)  # load your model
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    return train_set, model, optimizer


def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )


def ddp_module(save_every: int, total_epochs: int, batch_size: int, model,optimizer,dataset, criterion,snapshot_path: str,train_batch):
    ddp_setup()
    # train_data = dataset
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = Trainer(model, train_data, optimizer, save_every, snapshot_path,criterion,train_batch)
    trainer.train(total_epochs)
    destroy_process_group()


