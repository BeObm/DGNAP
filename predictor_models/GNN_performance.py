# -*- coding: utf-8 -*-


from torch_geometric.nn import MessagePassing
from predictor_models.utils import *
from torch_geometric.nn import global_add_pool  # global_mean_pool, global_max_pool,
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphSAGE
from torch_geometric.nn.norm import GraphNorm, InstanceNorm, BatchNorm

from settings.config_file import *
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Predictor(MessagePassing):
    def __init__(self, in_channels, dim, out_channels, drop_out):
        super(Predictor, self).__init__()
        #         self.embed_edges = Linear(self.edge_attr_size, self.hidden_channels)
        # accelerate.print("in channels dim =",in_channels)
        self.conv1 = GraphSAGE(in_channels,128,4, dim,)

        self.conv2 = GraphSAGE(dim,128,3, dim)
        self.drop_out = drop_out
        # self.normalize = InstanceNorm(dim)
        self.graphnorm = GraphNorm(dim)
        self.linear = Linear(dim, 64)
        self.linear2 = Linear(64, out_channels)

    def forward(self, data):
        x, edge_index, batch= data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = F.dropout(x, p=self.drop_out, training=self.training)

        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.drop_out, training=self.training)

        x = global_add_pool(x, batch)
        # x = self.graphnorm(x)

        x = F.relu(self.linear(x))
        x = self.linear2(x)

        return x


def train_predictor(model, dataloader, criterion, optimizer,accelerator):
    model.train()

    total_loss = 0
    for data in dataloader:
        targets=data.y
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output,targets)
        accelerator.backward(loss)
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
    return loss.item()


@torch.no_grad()
def test_predictor(accelerator, model, test_loader, metrics_list, title):
    model.eval()
    ped_list, label_list = [], []
    for data in test_loader:
        targets = data.y
        pred = model(data)

        all_targets = accelerator.gather(targets)
        all_pred = accelerator.gather(pred)

        ped_list = np.append(ped_list, all_pred.cpu().detach().numpy())
        label_list = np.append(label_list, all_targets.cpu().detach().numpy())
    predictor_performance = evaluate_model_predictor(label_list, ped_list, metrics_list, title)

    return predictor_performance
