import torch


class Cnn1DModel(torch.nn.Module):
    def __init__(
        self,
        classes,
        features=3328,
        hidden_dim=256,
        channels=(32, 64),
        conv_kernels=(5, 5),
        pooling_kernels=(3, 3),
        device="cpu",
    ):
        super(Cnn1DModel, self).__init__()
        # Layer 1
        self.conv1 = torch.nn.Conv1d(1, channels[0], conv_kernels[0], padding=1, device=device)
        self.relu1 = torch.nn.ReLU()
        self.maxpool1 = torch.nn.MaxPool1d(kernel_size=pooling_kernels[0])

        # Layer 2
        self.conv2 = torch.nn.Conv1d(channels[0], channels[1], conv_kernels[1], padding=1, device=device)
        self.relu2 = torch.nn.ReLU()
        self.maxpool2 = torch.nn.MaxPool1d(kernel_size=pooling_kernels[1])

        # Layer 3: Fully Connected
        self.fc1 = torch.nn.Linear(in_features=features, out_features=hidden_dim, device=device)
        self.relu3 = torch.nn.ReLU()

        # Layer 4: Fully Connected
        self.fc2 = torch.nn.Linear(in_features=hidden_dim, out_features=classes, device=device)
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = x.unsqueeze(1)

        # Layer 1
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        # Layer 2
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        # Layer 3
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)

        # Layer 4
        x = self.fc2(x)
        x = self.logsoftmax(x)
        return x
