class OneHotOperation:
    def __init__(self):
        self.value2index: dict = None

    '''
    def fit(self, f_values):
        self.value2index = dict()
        idx = 0
        for f_value in f_values:
            if str(f_value) in {'nan', 'None', '\\N'}:
                f_value = 'null'
            if f_value not in self.value2index:
                self.value2index[f_value] = idx
                idx += 1
    '''

    def fit(self, f_values):
        self.value2index = dict()
        filtered_f_values = list(filter(lambda x: str(x) not in {'', '\\N', 'null', 'Null', 'NULL', 'none', 'None', 'nan'}, f_values))
        filtered_f_values.sort(key=lambda x: x)
        if len(filtered_f_values) != len(f_values):
            self.value2index['null'] = 0
        idx = 1
        for f_value in filtered_f_values:
            if f_value not in self.value2index:
                self.value2index[f_value] = idx
                idx += 1

    def load(self, line):
        self.value2index = dict()
        for items in [tmp.split(":") for tmp in line.strip().split(',')]:
            self.value2index[items[0]] = int(items[1])

    def dump(self):
        return ",".join(str(x) + ":" + str(y) for x, y in self.value2index.items())

    def transform(self, f_value, offset):
        if str(f_value) in {'nan', 'None', '\\N'}:
            f_value = 'null'
        if f_value in self.value2index:
            return offset + self.value2index[f_value], 1.0
        else:
            return None, None

    def size(self):
        return 0 if self.value2index is None else len(self.value2index)