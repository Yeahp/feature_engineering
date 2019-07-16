import pandas as pd
import numpy as np
from feature_engineering.feature_comp.FeatureTransformer import FeatureTransformer
from feature_engineering.feature_comp.ChiSquareOneHotOperation import ChiSquareOneHotOperation


class FeatureInfo:
    def __init__(self):
        self.transformers: list = None
        self.names: list = None
        self.types: dict = None
        self.offsets: list = None

    def fit(self, feature_raw_path, data_path):
        self.transformers = list()
        self.names = list()
        self.types = dict()
        self.offsets = list()
        with open(feature_raw_path, 'r') as f:
            for line in f.readlines():
                transformer = FeatureTransformer()
                transformer.init(line)
                self.transformers.append(transformer)
                self.names.append(transformer.feature_name)
                self.types[transformer.feature_name] = np.float16 if transformer.feature_type == 'q' else np.str
        names = ['label']
        names.extend(self.names)
        self.types['label'] = np.str
        na_list = ['\\N', 'null', 'Null', 'NULL', 'none', 'None', 'nan']
        data = pd.read_table(data_path, sep='\t', header=None, names=names, dtype=self.types, na_values=na_list)
        for i in range(len(self.transformers)):
            if isinstance(self.transformers[i], ChiSquareOneHotOperation):
                self.transformers[i].fit(list(zip(data.ix[:, 0], data.ix[:, i + 1])))
                print('feature ' + str(i) + ': ' + self.names[i] + ' fit over!')
                if i == 0:
                    self.offsets.append(0)
                else:
                    self.offsets.append(self.offsets[i - 1] + self.transformers[i - 1].size())
            else:
                self.transformers[i].fit(data.ix[:, i + 1].tolist())
                print('feature ' + str(i) + ': ' + self.names[i] + ' fit over!')
                if i == 0:
                    self.offsets.append(0)
                else:
                    self.offsets.append(self.offsets[i - 1] + self.transformers[i - 1].size())

    def load(self, feature_transform_path):
        self.transformers = list()
        self.names = list()
        self.types = dict()
        self.offsets = list()
        with open(feature_transform_path, 'r') as f:
            lines = list(filter(lambda x: len(x.strip()) > 1, f.readlines()))
            for i in range(len(lines)):
                transformer = FeatureTransformer()
                transformer.load(lines[i])
                self.transformers.append(transformer)
                self.names.append(transformer.feature_name)
                self.types[transformer.feature_name] = np.float16 if transformer.feature_type == 'q' else np.str
                if i == 0:
                    self.offsets.append(0)
                else:
                    self.offsets.append(self.offsets[i - 1] + self.transformers[i - 1].size())

    def dump(self, feature_transform_path):
        with open(feature_transform_path, 'w') as f:
            for i in self.transformers:
                f.write(i.dump() + '\n')

    def transform(self, line, form):
        res = list()
        items = line.strip().split('\t')
        assert len(items) == (len(self.transformers) + 1)
        res.append(items[0])  # column 0 indicates label
        for i in range(len(items[1:])):
            feature_id, feature_value = self.transformers[i].transform(items[i + 1], self.offsets[i])
            if feature_id is not None and feature_value is not None:
                if form == 'ffm':
                    field_id = self.transformers[i].field_id
                    res.append(str(field_id) + ":" + str(feature_id) + ":" + str(feature_value))
                elif form == 'svm':
                    res.append(str(feature_id) + ":" + str(feature_value))
                else:
                    res.append(str(feature_id) + ":" + str(feature_value))
        return "\t".join(res)
