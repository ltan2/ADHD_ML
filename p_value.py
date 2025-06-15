import os
import glob
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif


def extract_with_pvalue(feature_filename, patient_info_filename, output_filename, act_or_hrv):
    features = pd.read_csv(feature_filename)
    features.fillna(0, inplace=True)
    features.replace([np.inf, -np.inf], 0, inplace=True)

    patient_df = pd.read_csv(patient_info_filename, delimiter=";", header=0)

    valid_patient_ids = []

    if act_or_hrv == "ACT":
        for patient_id in patient_df["ID"]:
            if patient_id < 10:
                filename = "patient_activity_0" + str(patient_id) + ".csv"
                if os.path.exists(f"dataset/activity_data/{filename}"):
                    valid_patient_ids.append(patient_id)
            else:
                filename = "patient_activity_" + str(patient_id) + ".csv"
                if os.path.exists(f"dataset/activity_data/{filename}"):
                    valid_patient_ids.append(patient_id)

    else:
        for patient_id in patient_df["ID"]:
            filename = "patient_hr" + str(patient_id) + ".csv"
            if os.path.exists(f"dataset/hrv_data/{filename}"):
                valid_patient_ids.append(patient_id)


    filtered_patient_df = patient_df[patient_df["ID"].isin(valid_patient_ids)]
    adhd_target = filtered_patient_df["ADHD"]

    # Perform ANOVA F-test for feature selection
    anova_selector = SelectKBest(score_func=f_classif, k='all')
    anova_selector.fit(features, adhd_target)

    p_values = pd.Series(anova_selector.pvalues_, index=features.columns)

    significant_features = p_values[p_values < 0.05].index.tolist()
    filtered_feature_df = features[significant_features]

    filtered_feature_df.to_csv(output_filename)
    