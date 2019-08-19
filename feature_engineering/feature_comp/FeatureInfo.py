import pandas as pd
import numpy as np
import threading
from feature_engineering.feature_comp.FeatureTransformer import FeatureTransformer
from feature_engineering.feature_comp.ChiSquareOneHotOperation import ChiSquareOneHotOperation


def _fit(data: pd.DataFrame, names: list, transformers: list, no: int):
    if isinstance(transformers[no].operation, ChiSquareOneHotOperation):
        transformers[no].fit(list(zip(data.ix[:, no + 1].tolist(), data.ix[:, 0].tolist())))
    else:
        transformers[no].fit(data.ix[:, no + 1].tolist())
    print('feature ' + str(no) + ': ' + names[no] + ' fit over!')


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
        data = pd.read_csv(data_path, sep='\t', header=None, names=names, dtype=self.types, na_values=na_list)
        threads = list()
        for i in range(len(self.transformers)):
            new_thread = threading.Thread(target=_fit(data=data, names=self.names, transformers=self.transformers, no=i))
            new_thread.start()
            threads.append(new_thread)
        for thread in threads:
            thread.join()
        for i in range(len(self.transformers)):
            '''
            if isinstance(self.transformers[i].operation, ChiSquareOneHotOperation):
                self.transformers[i].fit(list(zip(data.ix[:, i + 1].tolist(), data.ix[:, 0].tolist())))
            else :
                self.transformers[i].fit(data.ix[:, i + 1].tolist())
            print('feature ' + str(i) + ': ' + self.names[i] + ' fit over!')
            '''
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
