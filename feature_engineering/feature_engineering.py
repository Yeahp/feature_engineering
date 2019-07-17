from feature_engineering.feature_comp.FeatureInfo import FeatureInfo


class feature_engineering:

    @staticmethod
    def train_feature_info(origin_train_path: str, feature_path: str):
        feature_transform_path = feature_path + ".transform"
        feature_info = FeatureInfo()
        feature_info.fit(feature_path, origin_train_path)
        feature_info.dump(feature_transform_path)

    @staticmethod
    def feature_transform(feature_path: str, origin_data_path: str, new_data_path: str, sample_format: str):
        feature_transform_path = feature_path + ".transform"
        feature_info = FeatureInfo()
        feature_info.load(feature_transform_path)
        with open(origin_data_path, 'r') as f1, open(new_data_path, 'w') as f2:
            tmp = f1.readline()
            while tmp:
                f2.write(feature_info.transform(tmp.strip(), sample_format) + "\n")
                tmp = f1.readline()



