import argparse
import torch

from paifu_data import load_paifu_data, make_dataset, make_dataloader, train_model, test_model
from pai_const import code2hai

from cnn_model import CnnModel


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

model = CnnModel(len(code2hai), features=3392, hidden_dim=256, channels=(32, 64), kernel_sizes=(3, 3), device=device)
# model = CnnModel(len(code2hai), features=3328, hidden_dim = 256, channels=(32, 64), kernel_sizes=(5, 5), device=device)
# model = CnnModel(len(code2hai), features=3264, hidden_dim = 256, channels=(32, 64), kernel_sizes=(7, 7), device=device)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"# of parameters: {total_params}")

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

for epoch in range(1000):
    loss = train_model(model, criterion, optimizer, train_loader)
    acc = test_model(model, test_loader)
    print(f"epoch: {epoch}, loss: {loss}, acc: {acc}")
torch.save(model.state_dict(), "cnn_model.pth")