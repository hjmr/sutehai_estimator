import itertools
from pai_const import code2disp

sutehai_flags = {
    "tedashi": 0,
    "tsumogiri": 1,
    "naki": 2,
    "richi": 4,
}


class Player:
    def __init__(self, code, kyoku):
        self.code = code
        self.name = code
        self.tehai = []
        self.tsumo = 0
        self.furo = []
        self.sutehai = []
        self.sutehai_flags = []
        self.richi = False
        self.tsumogiri = False
        self.point = 0
        self.kaze = 0
        self.kyoku = kyoku

    def do_ripai(self):
        self.tehai.sort()

    def do_haipai(self, haipai):
        self.tehai.extend(haipai)
        self.do_ripai()

    def do_tsumo(self, tsumo):
        self.tsumo = tsumo
        # self.tehai.append(tsumo)
        self.tsumogiri = False

    def do_sutehai(self, sutehai, tsumogiri):
        self.sutehai.append(sutehai)
        if tsumogiri:
            self.sutehai_flags.append(sutehai_flags["tsumogiri"])
        else:
            if 0 < self.tsumo:
                self.tehai.append(self.tsumo)
            self.tehai.remove(sutehai)
            self.sutehai_flags.append(sutehai_flags["tedashi"])
        self.tsumo = 0
        self.tsumogiri = tsumogiri
        self.do_ripai()

    def do_richi(self):
        self.richi = True
        self.sutehai_flags[-1] += sutehai_flags["richi"]

    def do_open_kakan(self, tedashi, naki):
        if 0 < self.tsumo:
            self.tehai.append(self.tsumo)
        self.tsumo = 0
        self.tehai.remove(naki)
        self.furo.append(naki)

    def do_open_ankan(self, tedashi, naki):
        if 0 < self.tsumo:
            self.tehai.append(self.tsumo)
        self.tsumo = 0
        for hai in tedashi:
            self.tehai.remove(hai)
            self.furo.append(hai)

    def do_open_ponchi(self, tedashi, naki):
        for hai in tedashi:
            self.tehai.remove(hai)
            self.furo.append(hai)
        # self.kyoku.teban[-2].sutehai.pop() # remove the last sutehai
        self.kyoku.teban[-2].sutehai_flags[-1] += sutehai_flags["naki"]
        self.furo.append(naki)

    def show(self):
        def is_tsumogiri(flag):
            return 0 < (flag % 2)

        def is_naki(flag):
            return 0 < ((flag // 2) % 2)

        def is_richi(flag):
            return 0 < ((flag // 4) % 2)

        disp_tehai = [code2disp[self.tehai[idx] if idx < len(self.tehai) else 0] for idx in range(13)]
        disp_furo = [code2disp[self.furo[idx] if idx < len(self.furo) else 0] for idx in range(16)]
        # sutehai display
        # fmt: off
        disp_sutehai = [code2disp[self.sutehai[idx] if idx < len(self.sutehai) else 0] for idx in range(25)]
        disp_richi_flags     = [is_richi(self.sutehai_flags[idx])     if idx < len(self.sutehai_flags) else 0 for idx in range(25)]
        disp_naki_flags      = [is_naki(self.sutehai_flags[idx])      if idx < len(self.sutehai_flags) else 0 for idx in range(25)]
        disp_tsumogiri_flags = [is_tsumogiri(self.sutehai_flags[idx]) if idx < len(self.sutehai_flags) else 0 for idx in range(25)]
        # fmt: on
        disp_sutehai_flags = [
            "R" if r else "v" if n else "*" if t else " "
            for r, n, t in zip(disp_richi_flags, disp_naki_flags, disp_tsumogiri_flags)
        ]
        disp_sutehai = [hai + flag for hai, flag in zip(disp_sutehai, disp_sutehai_flags)]

        disp_rich = "R" if self.richi else " "
        disp_str = (
            "".join(disp_tehai)
            + "|"
            + code2disp[self.tsumo]
            + "|"
            + "".join(disp_furo)
            + "|"
            + "".join(disp_sutehai)
            + "|"
            + disp_rich
            + "|"
            + str(self.point)
        )
        print(f"{self.name:<10}\t" + ":" + disp_str)

    def _make_pai_data(self, lst, num, onehot=False):
        ret = []
        if onehot:
            for idx in range(num):
                v = [0] * (len(code2disp) - 1)
                if idx < len(lst):
                    v[lst[idx] - 1] = 1
                ret.extend(v)
        else:
            ret = [lst[idx] / (len(code2disp) - 1) if idx < len(lst) else 0 for idx in range(num)]
        return ret

    def _make_flags_data(self, lst, num, func, onehot=False):
        ret = []
        if onehot:
            ret = [[1, func(lst[idx])] if idx < len(lst) else [0, 0] for idx in range(num)]
            ret = list(itertools.chain.from_iterable(ret))
        else:
            ret = [(func(lst[idx]) + 1) / 2 if idx < len(lst) else 0 for idx in range(num)]
        return ret

    def get_tehai_data(self, onehot=False):
        return self._make_pai_data(self.tehai, 13, onehot)

    def get_tsumo_data(self, onehot=False):
        return self._make_pai_data([self.tsumo], 1, onehot)

    def get_furo_data(self, onehot=False):
        return self._make_pai_data(self.furo, 16, onehot)

    def get_sutehai_data(self, onehot=False):
        return self._make_pai_data(self.sutehai, 25, onehot)

    def get_tsumogiri_flags(self, onehot=False):
        def is_tsumogiri(flag):
            return flag % 2

        return self._make_flags_data(self.sutehai_flags, 25, is_tsumogiri, onehot)

    def get_naki_flags(self, onehot=False):
        def is_naki(flag):
            return (flag // 2) % 2

        return self._make_flags_data(self.sutehai_flags, 25, is_naki, onehot)

    def get_richi_flags(self, onehot=False):
        def is_richi(flag):
            return (flag // 4) % 2

        return self._make_flags_data(self.sutehai_flags, 25, is_richi, onehot)

    def __str__(self):
        # fmt: off
        str_array = [
            "tehai:",     str(self.tehai),
            "tsumo:",     str(self.tsumo),
            "furo:",      str(self.furo),
            "sutehai:",   str(self.sutehai),
            "richi:",     str(self.richi),
            "tsumogiri:", str(self.tsumogiri),
            "point:",     str(self.point),
            "kaze:",      str(self.kaze),
        ]
        # fmt: on
        return " ".join(str_array)
