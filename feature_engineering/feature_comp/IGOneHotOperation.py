import sys
import collections
import numpy as np


class IGOneHotOperation:

    def __init__(self, min_gain=0.05):
        self.min_gain: float = min_gain
        self.thresholds: list = None
        self.has_null: int = 0

    def fit(self, f_values):
        filtered_f_values = list(filter(lambda x: str(x[1]) not in {'', '\\N', 'null', 'Null', 'NULL', 'none', 'None', 'nan'}, f_values))
        if len(f_values) != len(filtered_f_values):
            self.has_null = 1

        filtered_f_values.sort(key=lambda x: (x[0], x[1]))
        fea_label_dic = {}
        for fea_label in filtered_f_values:
            fea_val = fea_label[0]
            label = fea_label[1]
            fea_label_dic.setdefault(fea_val, [0, 0])
            if label == '1':
                fea_label_dic[fea_val][0] += 1
            elif label == '0':
                fea_label_dic[fea_val][1] += 1
            else:
                print("data exception: label type error!")
                sys.exit(101)
        fea_label_list = [[key] + val for key, val in fea_label_dic.items()]

        left_index = 0
        right_index = 1
        self.thresholds.append(fea_label_list[left_index][0])
        while True:
            left_val = fea_label_list[left_index]
            right_val = fea_label_list[right_index]

            # calculate infomation gain
            pos_ratio = left_val[1] / (left_val[1] + left_val[2] + 0.0)
            neg_ratio = left_val[2] / (left_val[1] + left_val[2] + 0.0)
            if pos_ratio <= 0.0001 or neg_ratio <= 0.0001:
                left_gain = 0
            else:
                left_gain = -pos_ratio * np.log(pos_ratio) - neg_ratio * np.log(neg_ratio)
            pos_ratio = right_val[1] / (right_val[1] + right_val[2] + 0.0)
            neg_ratio = right_val[2] / (right_val[1] + right_val[2] + 0.0)
            if pos_ratio <= 0.0001 or neg_ratio <= 0.0001:
                right_gain = 0
            else:
                right_gain = -pos_ratio * np.log(pos_ratio) - neg_ratio * np.log(neg_ratio)

            pos_ratio = (left_val[1] + right_val[1]) / (left_val[1] + left_val[2] + right_val[1] + right_val[2] + 0.0)
            neg_ratio = (left_val[2] + right_val[2]) / (left_val[1] + left_val[2] + right_val[1] + right_val[2] + 0.0)
            if pos_ratio <= 0.0001 or neg_ratio <= 0.0001:
                all_gain = 0
            else:
                all_gain = -pos_ratio * np.log(pos_ratio) - neg_ratio * np.log(neg_ratio)

            # determine merge or not according to information gain
            if np.absolute(all_gain - left_gain - right_gain) <= self.min_gain:
                # concat the interval
                fea_label_list[left_index][1] += right_val[1]
                fea_label_list[left_index][2] += right_val[2]
                right_index += 1
            else:
                left_index = right_index
                right_index = left_index + 1
                self.thresholds.append(fea_label_list[left_index][0])

            if right_index >= len(fea_label_list):
                break

    def load(self, line):
        tokens = line.split('#')
        assert len(tokens) == 2
        self.thresholds = list() if len(tokens[0].strip()) == 0 else [float(x) for x in tokens[0].split(',')]
        self.has_null = 1 if tokens[1].strip() == '1' else 0

    def dump(self):
        return ",".join(str(x) for x in self.thresholds) + "#" + str(self.has_null)

    def transform(self, f_value, offset):
        if str(f_value) in {'', '\\N', 'null', 'Null', 'NULL', 'none', 'None', 'nan'}:
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
