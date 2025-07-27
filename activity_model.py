import os
from datetime import datetime
import csv
import pandas as pd
import tsfel
from utils import read_patient_info
from tsfresh import extract_features


def read_activity_file(patient_id):
    """
    Read patient activity file
    
    Args:
        patient_id: Identifier for the patient.

    Returns:
        pd.DataFrame: DataFrame containing timestamp, activity level, and patient ID.
    """
    data = []
    if patient_id < 10:
        filename = "patient_activity_0" + str(patient_id) + ".csv"
    else:
        filename = "patient_activity_" + str(patient_id) + ".csv"

    filepath = os.path.join(os.getcwd(), "dataset", "activity_data", filename) 

    try:
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                csv_reader = csv.reader(f, delimiter=";")
                next(csv_reader)  # Skip header
                for line in csv_reader:
                    timestamp = datetime.strptime(line[0], "%m-%d-%Y %H:%M")
                    activity = int(line[1].split(" ")[0])
                    data.append([timestamp, activity])
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        return pd.DataFrame()
    return pd.DataFrame(data, columns=["TIME", "ACT"]).assign(ID=patient_id)

def read_all_files_act(patient_file):
    """
    Process all patient activity files.
    
    Args:
        patient_ids: List of all patients

    Returns:
        pandas dataframe of from all patient
    """
    patient_df = read_patient_info(patient_file)
    all_data = []

    for patient_id in patient_df["ID"]:
        act_data = read_activity_file(patient_id)
        if not act_data.empty:
            act_data = extract_act_features(act_data, patient_id)
            adhd_label = patient_df.loc[patient_df["ID"] == patient_id, "ADHD"].values[0]
            act_data["ADHD"] = adhd_label
            all_data.append(act_data)

    return pd.concat(all_data, ignore_index=True)

def extract_act_features(patient_df, pid):
    cfg = tsfel.get_features_by_domain("statistical")
    time_diffs = patient_df['TIME'].diff().dt.total_seconds()
    median_diff = time_diffs.median()
    fs = 1.0 / median_diff if pd.notna(median_diff) and median_diff > 0 else 1.0
    
    signal_df = pd.DataFrame(patient_df['ACT'].values)
    
    feats = tsfel.time_series_features_extractor(cfg, signal_df, fs=fs, verbose=0)
    # feats = feats.loc[:, ~feats.columns.duplicated()]
    feats['ID'] = pid
    return feats
    
