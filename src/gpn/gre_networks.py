from torch import nn
import torch.nn.functional as F
import torch


def get_network(name):
    models = {
        "fc": FC_Batch_Net(7, 16, 8, 1),
    }
    return models[name.lower()]


def load_network(path, device):
    model_name = path.stem.split("_")[1]
    net = get_network(model_name).to(device)
    net.load_state_dict(torch.load(path, map_location=device))
    return net

class FC_Batch_Net(nn.Module):

    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(FC_Batch_Net, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.BatchNorm1d(n_hidden_1), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.BatchNorm1d(n_hidden_2), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        label_out = torch.sigmoid(self.layer3(x))
        label_out = label_out.squeeze(-1)
        return label_out