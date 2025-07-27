import os
import glob
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif


def extract_with_pvalue(features, adhd_target):
    features.fillna(0, inplace=True)
    features.replace([np.inf, -np.inf], 0, inplace=True)

    # Perform ANOVA F-test for feature selection
    anova_selector = SelectKBest(score_func=f_classif, k='all')
    anova_selector.fit(features, adhd_target)

    p_values = pd.Series(anova_selector.pvalues_, index=features.columns)

    significant_features = p_values[p_values < 0.05].index.tolist()
    filtered_feature_df = features[significant_features]

    print(len(filtered_feature_df.columns))
    return filtered_feature_df
    