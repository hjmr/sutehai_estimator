import torch


class LinearModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, device="cpu"):
        super(LinearModel, self).__init__()
        self.layer1 = torch.nn.Linear(input_dim, hidden_dim, device=device)
        self.layer2 = torch.nn.Linear(hidden_dim, hidden_dim, device=device)
        self.layer3 = torch.nn.Linear(hidden_dim, output_dim, device=device)
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, x):
        h1 = torch.relu(self.layer1(x))
        h2 = torch.relu(self.layer2(h1))
        o = self.logsoftmax(self.layer3(h2))
        return o
