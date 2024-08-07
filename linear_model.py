import argparse
import torch

from paifu import load_paifu_data, make_dataset, make_dataloader, train_model, test_model
from pai_const import code2hai


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

input_dim = len(input_data[0])
output_dim = len(code2hai)

model = LinearModel(input_dim, 700, output_dim, device=device)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"# of parameters: {total_params}")

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

for epoch in range(1000):
    loss = train_model(model, criterion, optimizer, train_loader)
    acc = test_model(model, test_loader)
    print(f"epoch: {epoch}, loss: {loss}, acc: {acc}")
torch.save(model.state_dict(), "linear_model.pth")
