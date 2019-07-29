from feature_engineering.chi_square.ChiSquareImp import ChiSquareImp


class ChiSquareOneHotOperation:

    def __init__(self, alpha=0.05):
        self.alpha: float = alpha
        self.thresholds: list = None
        self.has_null: int = 0

    def fit(self, f_values):
        filtered_f_values = list(filter(lambda x: str(x[1]) not in {'\\N', 'null', 'Null', 'NULL', 'none', 'None', 'nan'}, f_values))
        if len(f_values) != len(filtered_f_values):
            self.has_null = 1
        fea_label_cnt = ChiSquareImp.feature_label_count(filtered_f_values)
        fea_label_tuple = ChiSquareImp.build_cross_table(fea_label_cnt)
        self.thresholds = ChiSquareImp.chi_square_merge(fea_label_tuple, self.alpha)
        #for i in range(len(thresholds_raw)):
        #    if i == 0 or (thresholds_raw[i] - thresholds_raw[i - 1]) >= 0.000001:
        #        self.thresholds.append(thresholds_raw[i])

    def load(self, line):
        tokens = line.split('#')
        assert len(tokens) == 2
        self.thresholds = list() if len(tokens[0].strip()) == 0 else [float(x) for x in tokens[0].split(',')]
        self.has_null = 1 if tokens[1].strip() == '1' else 0

    def dump(self):
        return ",".join(str(x) for x in self.thresholds) + "#" + str(self.has_null)

    def transform(self, f_value, offset):
        if str(f_value) in {'\\N', 'null', 'Null', 'NULL', 'none', 'None', 'nan'}:
            if self.has_null == 1:
                return offset, 1.0
            else:
                return None, None
        if len(self.thresholds) == 0:
            return None, None
        if self.has_null == 1:
            offset += 1
        return offset + self.find(f_value), 1.0

    def find(self, v):
        for i in range(len(self.thresholds)):
            if float(v) < self.thresholds[i]:
                return i
        return len(self.thresholds)

    def size(self):
        res = 0
        if len(self.thresholds) != 0:
            res += len(self.thresholds) + 1
        if self.has_null == 1:
            res += 1
        return res

