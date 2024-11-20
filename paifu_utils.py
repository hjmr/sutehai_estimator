import json


def load_paifu(file):
    with open(file) as f:
        json_data = json.load(f)
    return json_data


def count_kyoku(json_data):
    cnt = 0
    for entry in json_data:
        if entry["cmd"] == "kyokustart":
            cnt += 1
    return cnt


def extract_player_names(json_data):
    player_names = {}
    for entry in json_data:
        if entry["cmd"] == "player":
            player_names[entry["args"][0]] = entry["args"][1]
    return player_names


def extract_one_kyoku(json_data, kyoku_num):
    max_kyoku_num = count_kyoku(json_data)
    if kyoku_num < 0:
        raise ValueError(f"kyoku_num: {kyoku_num} must be larger than 0.")
    elif max_kyoku_num <= kyoku_num:
        raise ValueError(f"kyoku_num: {kyoku_num} is too large.")

    kyoku_num += 1
    kyoku = []
    for entry in json_data:
        if entry["cmd"] == "kyokustart":
            kyoku_num -= 1
            if kyoku_num == 0:
                kyoku.append(entry)
        elif kyoku_num == 0:
            kyoku.append(entry)
            if entry["cmd"] == "kyokuend":
                break
    return kyoku
