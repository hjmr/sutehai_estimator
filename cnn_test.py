import argparse
import torch

from cnn_model import CnnModel

from kyoku import Kyoku
from paifu_utils import load_paifu, extract_one_kyoku
from paifu_data import make_data_for_one_kyoku, make_dataset, make_dataloader
from pai_const import code2hai, code2disp


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", type=str, help="device")
    parser.add_argument("--paifu", type=str, help="paifu file")
    parser.add_argument("--kyoku", type=int, default=0, help="kyoku index")
    parser.add_argument("model", help="model file")
    return parser.parse_args()


args = parse_args()

torch.device("cpu")
if args.dev is not None:
    device = torch.device(args.dev)
elif torch.backends.mps.is_available():
    device = torch.device("mps")
print(f"device: {device}")

# load paifu data
json_data = load_paifu(args.paifu)
kyoku_data = extract_one_kyoku(json_data, args.kyoku)
inp, tgt, stps = make_data_for_one_kyoku(kyoku_data)
dataset = make_dataset(inp, tgt, device=device)

# prepare kyoku for display
kyoku = Kyoku(kyoku_data)

# load trained model
model = CnnModel(len(code2hai), features=3392, hidden_dim=256, channels=(32, 64), kernel_sizes=(3, 3), device=device)
# model = CnnModel(len(code2hai), features=3328, hidden_dim = 256, channels=(32, 64), kernel_sizes=(5, 5), device=device)
# model = CnnModel(len(code2hai), features=3264, hidden_dim = 256, channels=(32, 64), kernel_sizes=(7, 7), device=device)
model.load_state_dict(torch.load(args.model))

model.eval()
with torch.no_grad():
    for idx, (input, target) in enumerate(dataset):
        input = input.view(1, -1)
        outputs = model(input)
        _, predicted = torch.max(outputs, 1)
        print("--------------------------------------------------")
        kyoku.fast_forward(stps[idx])
        kyoku.show()
        judge = "[OK]" if target.item() == predicted.item() else "*NG*"
        print(f"{judge} target:{code2disp[target.item()]} predict:{code2disp[predicted.item()]}")
