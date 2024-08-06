from player import Player
from pai_const import code2hai, code2disp


class Kyoku:
    def __init__(self, kyoku_data: list):
        self.player_names = ["A0", "B0", "C0", "D0"]
        self.players = {}
        for player_name in self.player_names:
            self.players[player_name] = Player(player_name, self)

        self.oya = None
        self.dora = []
        self.honba = 0
        self.bakaze = 0
        self.ryukyoku = False

        self.kyoku_data = kyoku_data
        self.current_step = 0
        self.teban = []

        self.is_tsumo = False
        self.is_sutehai = False

        # fmt: off
        self.commands = {
            "haipai":     self.do_haipai,
            "ryukyoku":   self.do_ryukyoku,
            "dice":       self.do_dummy,
            "sutehai":    self.do_sutehai,
            "kyokustart": self.do_kyokustart,
            "tsumo":      self.do_tsumo,
            "point":      self.do_point,
            "dora":       self.do_dora,
            "open":       self.do_open,
            "kyokuend":   self.do_kyokuend,
            "say":        self.do_dummy,
            "richi":      self.do_richi,
            "uradora":    self.do_dora,
            "agari":      self.do_dummy,
        }

    # fmt: on

    def get_player(self, name):
        player = self.players[name]
        if len(self.teban) == 0 or self.teban[-1] != player:
            self.teban.append(player)
        return player

    def step(self):
        playing = True
        self.is_sutehai = False
        self.is_tsumo = False

        entry = self.kyoku_data[self.current_step]
        if entry["cmd"] not in self.commands:
            raise ValueError(f"Invalid command: {entry['cmd']}")
        playing = self.commands[entry["cmd"]](entry["args"])
        self.current_step += 1
        return playing

    def do_dummy(self, args):
        return True

    def do_kyokustart(self, args):
        self.oya = self.players[args[1]]
        self.honba = args[2]
        self.bakaze = code2hai.index(args[4])
        for idx in range(4):
            self.players[self.player_names[idx]].kaze = code2hai.index(args[5:][idx])
        return True

    def do_kyokuend(self, args):
        return False

    def do_ryukyoku(self, args):
        self.ryukyoku = True
        return True

    def do_haipai(self, args):
        player = self.get_player(args[0])
        haipai_str = args[1]
        haipai = [code2hai.index(haipai_str[idx : idx + 2]) for idx in range(0, len(haipai_str), 2)]
        player.do_haipai(haipai)
        return True

    def do_tsumo(self, args):
        player = self.get_player(args[0])
        tsumo_code = code2hai.index(args[2])
        player.do_tsumo(tsumo_code)
        self.is_tsumo = True
        return True

    def do_sutehai(self, args):
        player = self.get_player(args[0])
        sutehai_code = code2hai.index(args[1])
        tsumogiri = True if len(args) == 3 and args[2] == "tsumogiri" else False
        player.do_sutehai(sutehai_code, tsumogiri)
        self.is_sutehai = True
        return True

    def do_dora(self, args):
        if args[1] in code2hai:
            dora_code = code2hai.index(args[1])
            self.dora.append(dora_code)
        return True

    def do_open(self, args):
        open_flag = args[1][0]
        if open_flag not in ["[", "(", "<"]:
            return True

        player = self.get_player(args[0])
        open_funcs = {"[": player.do_open_kakan, "(": player.do_open_ankan, "<": player.do_open_ponchi}
        tedashi_str = args[1][1:-1]
        tedashi_code = [code2hai.index(tedashi_str[idx : idx + 2]) for idx in range(0, len(tedashi_str), 2)]
        naki_code = code2hai.index(args[2]) if len(args) == 3 else 0
        open_funcs[open_flag](tedashi_code, naki_code)
        return True

    def do_richi(self, args):
        self.players[args[0]].do_richi()
        return True

    def do_point(self, args):
        player = self.players[args[0]]
        point_op = args[1][0]
        if point_op == "+":
            player.point += int(args[1][1:])
        elif point_op == "-":
            player.point -= int(args[1][1:])
        elif point_op == "=":
            player.point = int(args[1][1:])
        elif "0" <= point_op and point_op <= "9":
            if self.ryukyoku:
                player.point += int(args[1])
            else:
                player.point = int(args[1])
        else:
            raise ValueError("Invalid point operation")
        return True

    def show(self):
        disp_dora = "".join([code2disp[dora] for dora in self.dora])
        if 0 < len(self.teban):
            print("teban: " + self.teban[-1].name + " dora: " + disp_dora)
        for player_name in self.player_names:
            self.players[player_name].show()

    def get_data(self):
        def nlbs(lst, size):  # normalize list by size
            return [v / size for v in lst]

        current_data = []

        if 0 < len(self.teban):
            dora_data = [self.dora[idx] if idx < len(self.dora) else 0 for idx in range(4)]
            current_data.extend(nlbs(dora_data, len(code2hai) - 1))

            teban_idx = self.player_names.index(self.teban[-1].name)
            for rel_idx in range(4):
                idx = (teban_idx + rel_idx) % 4
                player_name = self.player_names[idx]
                player = self.players[player_name]

                # 手番のプレイヤーの手牌
                if rel_idx == 0:  # teban
                    current_data.extend(nlbs(player.get_tehai_data(), len(code2hai) - 1))
                    current_data.append(player.get_tsumo_data() / (len(code2hai) - 1))

                current_data.extend(nlbs(player.get_furo_data(), len(code2hai) - 1))
                current_data.extend(nlbs(player.get_sutehai_data(), len(code2hai) - 1))
                current_data.extend(nlbs(player.get_tsumogiri_flags(), 2))
                current_data.extend(nlbs(player.get_richi_flags(), 2))
                current_data.extend(nlbs(player.get_naki_flags(), 2))

        return current_data
