import sys
from feature_engineering.feature_comp.OneHotOperation import OneHotOperation
from feature_engineering.feature_comp.CdfOneHotOperation import CdfOneHotOperation
from feature_engineering.feature_comp.ChiSquareOneHotOperation import ChiSquareOneHotOperation
from feature_engineering.feature_comp.IGOneHotOperation import IGOneHotOperation


class FeatureTransformer:

    def __init__(self):
        self.line_id: int = None
        self.feature_name: str = None
        self.feature_type: str = None
        self.field_id: int = None
        self.transform_info: str = None
        self.operation = None

    def init(self, feature_info):
        items = feature_info.strip().split('\t')
        assert len(items) >= 3
        self.line_id = int(items[0])
        self.feature_name = items[1]
        self.feature_type = items[2]
        self.field_id = self.line_id
        self.transform_info = None
        self.operation = None
        if len(items) >= 5:
            self.field_id = int(items[3])
            self.transform_info = items[4]
            tokens = self.transform_info.split('#')
            transformer_type = tokens[0]
            if transformer_type == 'cdf_onehot':
                assert len(tokens) == 2
                self.operation = CdfOneHotOperation(int(tokens[1]))
            elif transformer_type == 'chi_onehot':
                assert len(tokens) == 2
                self.operation = ChiSquareOneHotOperation(float(tokens[1]))
            elif transformer_type == 'ig_onehot':
                assert len(tokens) == 2
                self.operation = IGOneHotOperation(float(tokens[1]))
            elif transformer_type == 'onehot':
                self.operation = OneHotOperation()
            else:
                print('unkown transformer type: ' + transformer_type)
                sys.exit(404)

    def fit(self, f_values):
        if self.operation is not None:
            self.operation.fit(f_values)

    def load(self, line):
        items = line.strip().split('\t')
        assert len(items) >= 3
        self.line_id = int(items[0])
        self.feature_name = items[1]
        self.feature_type = items[2]
        self.field_id = self.line_id
        if len(items) >= 6:
            self.field_id = int(items[3])
            self.transform_info = items[4]
            tokens = self.transform_info.split('#')
            transformer_type = tokens[0]
            if transformer_type == 'cdf_onehot':
                assert len(tokens) == 2
                self.operation = CdfOneHotOperation(int(tokens[1]))
                self.operation.load(items[5])
            elif transformer_type == 'chi_onehot':
                assert len(tokens) == 2
                self.operation = ChiSquareOneHotOperation(float(tokens[1]))
                self.operation.load(items[5])
            elif transformer_type == 'ig_onehot':
                assert len(tokens) == 2
                self.operation = IGOneHotOperation(float(tokens[1]))
                self.operation.load(items[5])
            elif transformer_type == 'onehot':
                self.operation = OneHotOperation()
                self.operation.load(items[5])
            else:
                print('unkown transformer type: ' + transformer_type)
                sys.exit(404)

    def dump(self):
        items = list()
        items.append(str(self.line_id))
        items.append(self.feature_name)
        items.append(self.feature_type)
        items.append(str(self.field_id))
        if self.transform_info is not None and self.operation is not None:
            items.append(self.transform_info)
            items.append(self.operation.dump())
        return "\t".join(items)

    def transform(self, f_value, offset):
        if self.operation is None:
            return offset, f_value
        return self.operation.transform(f_value, offset)

    def size(self):
        return 1 if self.operation is None else self.operation.size()
