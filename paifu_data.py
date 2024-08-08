import argparse
import torch
import torch.utils

from kyoku import Kyoku
from paifu_utils import load_paifu, count_kyoku, extract_one_kyoku


def make_data_for_one_kyoku(kyoku_json):
    kyoku = Kyoku(kyoku_json)
    stps = []
    inp = []
    tgt = []
    while kyoku.step():
        if kyoku.was_sutehai:
            tgt.append(kyoku.teban[-1].sutehai[-1])
        if kyoku.will_sutehai():
            stps.append(kyoku.current_step)
            inp.append(kyoku.get_data())
    return inp, tgt, stps


def load_paifu_data(files):
    kyoku_steps = []
    input_data = []
    target_data = []

    for f in files:
        json_data = load_paifu(f)
        kyoku_num = count_kyoku(json_data)
        for idx in range(kyoku_num):
            kyoku_json = extract_one_kyoku(json_data, idx)
            inp, tgt, stp = make_data_for_one_kyoku(kyoku_json)
            kyoku_steps.extend(stp)
            input_data.extend(inp)
            target_data.extend(tgt)
    return input_data, target_data, kyoku_steps


def make_dataset(input_data, target_data, device="cpu"):
    input = torch.tensor(input_data, dtype=torch.float32, device=device)
    target = torch.tensor(target_data, dtype=torch.long, device=device)
    return torch.utils.data.TensorDataset(input, target)


def make_dataloader(dataset, batch_size, train_ratio=0.8):
    train_num = int(len(dataset) * train_ratio)
    test_num = len(dataset) - train_num
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_num, test_num])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def train_model(model, criterion, optimizer, train_loader):
    model.train()
    for _, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    return loss.item()


def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    return correct / total
