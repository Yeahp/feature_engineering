import sys
from feature_engineering.feature_comp.FeatureInfo import FeatureInfo


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("argument error!")
        sys.exit(0)
    originTrainDataPath = sys.argv[1]
    featurePath = sys.argv[2]
    featureTransformPath = featurePath + ".transform"
    featureInfo = FeatureInfo()
    featureInfo.fit(featurePath, originTrainDataPath)
    featureInfo.dump(featureTransformPath)
