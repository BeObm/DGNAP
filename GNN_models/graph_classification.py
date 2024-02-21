# -*- coding: utf-8 -*-


from torch.nn import Linear
from torch_geometric.nn.norm import GraphNorm
# from torch_geometric.nn.models import MLP
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
import inspect
from torch.nn import Linear, ReLU, Sequential
from search_algo.utils import *
# from sklearn.metrics import accuracy_score#, precision_score, recall_score
from settings.config_file import *



class GNN_Model(MessagePassing):
    def __init__(self, param_dict):
        super(GNN_Model, self).__init__()
        self.in_feat=param_dict["in_channels"]
        self.num_class =param_dict["num_class"]
        self.aggr1 = param_dict['aggregation1']
        self.aggr2 = param_dict['aggregation2']
        self.hidden_channels = int(param_dict['hidden_channels'])
        self.type_task = config["dataset"]["type_task"]
        self.head1 = 8
        self.head2 =8
        self.gnnConv1 = param_dict['gnnConv1'][0]
        self.gnnConv2 = param_dict['gnnConv2'][0]
        self.global_pooling = param_dict['pooling']
        self.activation1 = param_dict['activation1']
        self.activation2 = param_dict['activation2']
        self.dropout = param_dict['dropout']
        self.normalize1 = param_dict['normalize1']
        self.normalize2 = param_dict['normalize2']

        if param_dict['gnnConv1'][1] == "linear":
            self.conv1 = self.gnnConv1(self.in_feat, self.hidden_channels)
            self.input_conv2 = self.hidden_channels
        else:
            if 'head' in inspect.getfullargspec(self.gnnConv1)[0]:
                self.conv1 = self.gnnConv1(self.in_feat, self.hidden_channels, head=self.head1,
                                           aggr=self.aggr1)
                self.input_conv2 = self.hidden_channels * self.head1
            elif 'heads' in inspect.getfullargspec(self.gnnConv1)[0]:
                self.conv1 = self.gnnConv1(self.in_feat, self.hidden_channels, heads=self.head1,
                                           aggr=self.aggr1)
                self.input_conv2 = self.hidden_channels * self.head1
            elif 'num_heads' in inspect.getfullargspec(self.gnnConv1)[0]:
                self.conv1 = self.gnnConv1(self.in_feat, self.hidden_channels, num_heads=self.head1,
                                           aggr=self.aggr1)
                self.input_conv2 = self.hidden_channels * self.head1

            else:
                if param_dict['gnnConv1'][1] == "SGConv":  # splineconv does not support multihead
                    self.conv1 = self.gnnConv1(self.in_feat, self.hidden_channels, K=2,
                                               aggr=self.aggr1)
                else:
                    self.conv1 = self.gnnConv1(self.in_feat, self.hidden_channels)

                self.input_conv2 = self.hidden_channels

        if param_dict['gnnConv2'][1] == "linear":
            self.conv2 = self.gnnConv2(self.input_conv2, self.hidden_channels)
            self.output_conv2 = self.hidden_channels
        else:
            if 'head' in inspect.getfullargspec(self.gnnConv2)[0]:
                self.conv2 = self.gnnConv2(self.input_conv2, self.hidden_channels, head=self.head2, aggr=self.aggr2)
                self.output_conv2 = self.hidden_channels * self.head2
            elif 'heads' in inspect.getfullargspec(self.gnnConv2)[0]:
                self.conv2 = self.gnnConv2(self.input_conv2, self.hidden_channels, heads=self.head2, aggr=self.aggr2)
                self.output_conv2 = self.hidden_channels * self.head2
            elif 'num_heads' in inspect.getfullargspec(self.gnnConv2)[0]:
                self.conv2 = self.gnnConv2(self.input_conv2, self.hidden_channels, num_heads=self.head2,
                                           aggr=self.aggr2)
                self.output_conv2 = self.hidden_channels * self.head2
            else:
                if param_dict['gnnConv2'][1] == "SGConv":  # splineconv does not support multihead
                    self.conv2 = self.gnnConv2(self.input_conv2, self.hidden_channels, K=2, aggr=self.aggr2)
                else:
                    self.conv2 = self.gnnConv2(self.input_conv2, self.hidden_channels, aggr=self.aggr2)
                self.output_conv2 = self.hidden_channels

        if self.normalize1 != False:
            self.normalizer_function1 = self.normalize1(self.input_conv2)

        if self.normalize2 != False:
            self.normalizer_function2 = self.normalize2(self.output_conv2)

        self.mlp = Sequential(Linear(self.output_conv2, self.hidden_channels*16), ReLU(),
                              Linear(self.hidden_channels*16, self.hidden_channels), ReLU(),
                              Linear(self.hidden_channels, self.num_class))

    def get_forward_conv(self, num_layer, conv, x, edge_index, edge_attr):

        if ('edge_attr' in inspect.getfullargspec(conv.forward)[0]) or (
                "hypergraph_atrr" in inspect.getfullargspec(conv.forward)[0]) or (
                "edge_index" in inspect.getfullargspec(conv.forward)[0]):

            if num_layer == 1:
                return self.conv1(x, edge_index, edge_attr)
            if num_layer == 2:
                return self.conv2(x, edge_index, edge_attr)

        else:
            if num_layer == 1:
                return self.conv1(x, edge_index)
            if num_layer == 2:
                return self.conv2(x, edge_index)

    def forward(self, data):

        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        if 'batch' in data.keys:
            batch = data.batch

        #x = F.dropout(x, self.dropout, training=self.training)

        x = self.get_forward_conv(1, self.gnnConv1, x, edge_index, edge_attr)

        if self.normalize1 not in [False, "False"]:
            x = self.normalizer_function1(x)
        x = self.activation1(x)

        #x = F.dropout(x, self.dropout, training=self.training)
        x = self.get_forward_conv(2, self.gnnConv2, x, edge_index, edge_attr)

        if self.normalize2 not in [False, "False"]:
            x = self.normalizer_function2(x)
        x = self.activation2(x)


        # 2. Readout layer

        x = self.global_pooling(x, batch)  # [batch_size, self.hidden_channels]
        x = self.activation2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        #  Apply a final classifier
        x = self.mlp(x)

        x = F.log_softmax(x, dim=1)
        return x


def train_function(model, dataloader, criterion, optimizer,accelerator):
    model.train()
    loss_all = 0
    for batch in dataloader:
        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output, batch.y)
        accelerator.backward(loss)
        loss_all += batch.num_graphs * loss.item()
        optimizer.step()
    return loss_all / int(config["dataset"]["len_traindata"])


@torch.no_grad()
def test_function(accelerator, model, test_loader, type_data="val"):
    model.eval()
    true_labels = []
    pred_labels = []

    for data in test_loader:  # Iterate in batches over the training/test dataset.
        targets= data.y
        pred = model(data).max(dim=1)[1]
        all_targets =accelerator.gather(targets)
        all_pred = accelerator.gather(pred)
        true_labels.extend(all_targets.detach().cpu().numpy())
        pred_labels.extend(all_pred.detach().cpu().numpy())
    performance_scores = evaluate_model(true_labels, pred_labels,type_data)

    return performance_scores



