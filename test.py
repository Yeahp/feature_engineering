from feature_engineering.feature_engineering import feature_engineering


if __name__ == "__main__":
    origin_train_path = '/Users/hello/PycharmProjects/feature_engineering/feature_engineering/test/origin_train'
    new_data_path = '/Users/hello/PycharmProjects/feature_engineering/feature_engineering/test/new_train'
    feature_path = '/Users/hello/PycharmProjects/feature_engineering/feature_engineering/test/feature'
    feature_engineering.train_feature_info(origin_train_path, feature_path)
    feature_engineering.feature_transform(feature_path, origin_train_path, new_data_path, 'ffm')
