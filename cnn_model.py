import argparse
import torch

from paifu import load_paifu_data, make_dataset, make_dataloader, train_model, test_model
from pai_const import code2hai


class CnnModel(torch.nn.Module):
    def __init__(self, classes, device="cpu"):
        super(CnnModel, self).__init__()
        # Layer 1
        self.conv1 = torch.nn.Conv1d(in_channels=1, out_channels=32, kernel_size=7, padding=1, device=device)
        self.relu1 = torch.nn.ReLU()
        self.maxpool1 = torch.nn.MaxPool1d(kernel_size=3)

        # Layer 2
        self.conv2 = torch.nn.Conv1d(in_channels=32, out_channels=64, kernel_size=7, padding=1, device=device)
        self.relu2 = torch.nn.ReLU()
        self.maxpool2 = torch.nn.MaxPool1d(kernel_size=3)

        # Layer 3: Fully Connected
        self.fc1 = torch.nn.Linear(in_features=3264, out_features=256, device=device)
        self.relu3 = torch.nn.ReLU()

        # Layer 4: Fully Connected
        self.fc2 = torch.nn.Linear(256, classes, device=device)
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--dev", type=str, help="device")
    parser.add_argument("files", nargs="+", help="paifu files")
    return parser.parse_args()


args = parse_args()

torch.device("cpu")
if args.dev is not None:
    device = torch.device(args.dev)
elif torch.backends.mps.is_available():
    device = torch.device("mps")
print(f"device: {device}")

input_data, target_data = load_paifu_data(args.files)
dataset = make_dataset(input_data, target_data, device=device)
train_loader, test_loader = make_dataloader(dataset, batch_size=args.batch_size)

model = CnnModel(len(code2hai), device=device)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"# of parameters: {total_params}")

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

for epoch in range(1000):
    loss = train_model(model, criterion, optimizer, train_loader)
    acc = test_model(model, test_loader)
    print(f"epoch: {epoch}, loss: {loss}, acc: {acc}")
torch.save(model.state_dict(), "cnn_model.pth")
