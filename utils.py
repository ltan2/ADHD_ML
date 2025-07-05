import os
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def plot_activity(extracted_features):
    sns.boxplot(data=extracted_features, orient="h")
    plt.title("Feature Distribution Across Patients")
    plt.show()

def read_patient_info(filename):
    filepath = os.path.join(os.getcwd(), "dataset", filename)
    df = pd.read_csv(filepath, delimiter= ";", header = 0)
    return df

def read_X_and_Y():
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