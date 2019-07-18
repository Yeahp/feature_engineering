**Feature Engineering: One-Hot skills**

(1) Definition: One-Hot

    When doing some feature engineering work, we want to map feature values to some finite categories. 
    Once a feature is mapped to one of these categories, this category will be set 1 while the others 0. 

(2) What skills do we have?

    a. the common one-hot
       Each of feature values represents a category.
    b. the CDF one-hot
       CDF means cumulative distribution function here, while feature values are continuous and can
       be divided into several buckets, each of which can be regarded as a category.
    c. the Chi-Square one-hot
       The formal two skills treat each feature independently, that may not be suitable for supervised
        machine learning. Chi-Square one-hot takes both feature value and label into consideration. Given
        a label, Chi-Square test displays difference between observed groups which enable us to divide the 
        interval artificially and objectively for each feature.

(3) Demo: how to use this package?

    Step 1: install this package
    Step 2: write a script or adopt an interactive mode
    
    Here's an example: 
    
    from feature_engineering.feature_engineering import feature_engineering
    if __name__ == "__main__":
        origin_train_path = ''
        new_data_path = ''
        feature_path = ''
        feature_engineering.train_feature_info(origin_train_path, feature_path)
        feature_engineering.feature_tranform(feature_path, origin_train_path, new_data_path, 'ffm')
    