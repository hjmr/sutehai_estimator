from player import Player
from pai_const import code2pai, code2disp


class Kyoku:
    def __init__(self, kyoku_data: list, player_names: dict = None):
        self.player_codes = ["A0", "B0", "C0", "D0"]
        self.players = {}
        for p_code in self.player_codes:
            self.players[p_code] = Player(p_code, self)
            if player_names is not None:
                self.players[p_code].name = player_names[p_code]

        self.oya = None
        self.dora = []
        self.honba = 0
        self.bakaze = 0
        self.ryukyoku = False

        self.kyoku_data = kyoku_data
        self.current_step = 0
        self.teban = []

        self.was_tsumo = False
        self.was_sutehai = False

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

    def get_player(self, p_code):
        player = self.players[p_code]
        if len(self.teban) == 0 or self.teban[-1] != player:
            self.teban.append(player)
        return player

    def step(self):
        playing = True
        self.was_sutehai = False
        self.was_tsumo = False

        entry = self.kyoku_data[self.current_step]
        if entry["cmd"] not in self.commands:
            raise ValueError(f"Invalid command: {entry['cmd']}")
        playing = self.commands[entry["cmd"]](entry["args"])
        self.current_step += 1
        return playing

    def will_sutehai(self):
        return self.kyoku_data[self.current_step]["cmd"] == "sutehai"

    def fast_forward(self, steps):
        while self.current_step < steps:
            self.step()

    def do_dummy(self, args):
        return True

    def do_kyokustart(self, args):
        self.oya = self.players[args[1]]
        self.honba = args[2]
        self.bakaze = code2pai.index(args[4])
        for idx in range(4):
            self.players[self.player_codes[idx]].kaze = code2pai.index(args[5:][idx])
        return True

    def do_kyokuend(self, args):
        return False

    def do_ryukyoku(self, args):
        self.ryukyoku = True
        return True

    def do_haipai(self, args):
        player = self.get_player(args[0])
        haipai_str = args[1]
        haipai = [code2pai.index(haipai_str[idx : idx + 2]) for idx in range(0, len(haipai_str), 2)]
        player.do_haipai(haipai)
        return True

    def do_tsumo(self, args):
        player = self.get_player(args[0])
        tsumo_code = code2pai.index(args[2])
        player.do_tsumo(tsumo_code)
        self.was_tsumo = True
        return True

    def do_sutehai(self, args):
        player = self.get_player(args[0])
        sutehai_code = code2pai.index(args[1])
        tsumogiri = True if len(args) == 3 and args[2] == "tsumogiri" else False
        player.do_sutehai(sutehai_code, tsumogiri)
        self.was_sutehai = True
        return True

    def do_dora(self, args):
        if args[1] in code2pai:
            dora_code = code2pai.index(args[1])
            self.dora.append(dora_code)
        return True

    def do_open(self, args):
        open_flag = args[1][0]
        if open_flag not in ["[", "(", "<"]:
            return True

        player = self.get_player(args[0])
        open_funcs = {"[": player.do_open_kakan, "(": player.do_open_ankan, "<": player.do_open_ponchi}
        tedashi_str = args[1][1:-1]
        tedashi_code = [code2pai.index(tedashi_str[idx : idx + 2]) for idx in range(0, len(tedashi_str), 2)]
        naki_code = code2pai.index(args[2]) if len(args) == 3 else 0
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
        for p_code in self.player_codes:
            self.players[p_code].show()

    def get_data(self, onehot=False):
        current_data = []

        if 0 < len(self.teban):
            dora_data = []
            if onehot:
                for idx in range(4):
                    dora = [0] * (len(code2pai) - 1)
                    if idx < len(self.dora):
                        dora[self.dora[idx] - 1] = 1
                    dora_data.extend(dora)
            else:
                dora_data = [self.dora[idx] if idx < len(self.dora) else 0 for idx in range(4)]
            current_data.extend(dora_data)

            teban_idx = self.player_codes.index(self.teban[-1].code)
            for rel_idx in range(4):
                idx = (teban_idx + rel_idx) % 4
                p_code = self.player_codes[idx]
                player = self.players[p_code]

                # 手番のプレイヤーの手牌
                if rel_idx == 0:  # teban
                    current_data.extend(player.get_tehai_data(onehot))
                    current_data.extend(player.get_tsumo_data(onehot))

                current_data.extend(player.get_furo_data(onehot))
                current_data.extend(player.get_sutehai_data(onehot))
                current_data.extend(player.get_tsumogiri_flags(onehot))
                current_data.extend(player.get_richi_flags(onehot))
                current_data.extend(player.get_naki_flags(onehot))

        return current_data
