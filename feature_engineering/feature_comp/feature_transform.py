import sys
from feature_engineering.feature_comp.FeatureInfo import FeatureInfo


def do_transform(feature_info, origin_path, new_path, sample_format):
    with open(origin_path, 'r') as f1, open(new_path, 'w') as f2:
        tmp = f1.readline()
        while tmp:
            f2.write(feature_info.transform(tmp.strip(), sample_format) + "\n")
            tmp = f1.readline()


if __name__ == "__main__":
    if len(sys.argv) < 7:
        print("argument error!")
        sys.exit(0)
    originTrainDataPath = sys.argv[1]
    originTestDataPath = sys.argv[2]
    trainDataPath = sys.argv[3]
    testDataPath = sys.argv[4]
    featurePath = sys.argv[5]
    sampleFormat = sys.argv[6]
    featureTransformPath = featurePath + ".transform"
    featureInfo = FeatureInfo()
    featureInfo.load(featureTransformPath)
    do_transform(featureInfo, originTrainDataPath, trainDataPath, sampleFormat)
    do_transform(featureInfo, originTestDataPath, testDataPath, sampleFormat)
