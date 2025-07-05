import os
from datetime import datetime
import csv
import pandas as pd
import tsfel
from utils import read_patient_info
from tsfresh import extract_features


def read_hrv_file(patient_id):
    data = []
    filename = "patient_hr_" + str(patient_id) + ".csv"
    filepath = os.path.join(os.getcwd(), "dataset", "hrv_data", filename)

    try:
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                csv_reader = csv.reader(f, delimiter=";")
                next(csv_reader)  # Skip header
                for line in csv_reader:
                    timestamp = datetime.strptime(line[0], "%Y-%m-%d %H:%M:%S.%f")
                    activity = float(line[1].split(" ")[0])
                    data.append([timestamp, activity])
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        return pd.DataFrame()
    patient_df = pd.DataFrame(data, columns=["TIME", "HRV"]).assign(ID=patient_id)
    # extracted_features = tsfel.time_series_features_extractor(cfg, patient_df["HRV"], verbose=0)
    # extracted_features = extract_features(patient_df, column_id="ID", column_value="HRV", column_sort="TIME", n_jobs=2, show_warnings=False)
    return patient_df
    

def read_all_files_hrv(patient_file):
    """
    Process all patient hrv files.
    
    Args:
        patient_ids: List of all patients

    Returns:
        pandas dataframe of from all patient
    """
    patient_df = read_patient_info(patient_file)
    all_data = []

    for patient_id in patient_df["ID"]:
        hrv_data = read_hrv_file(patient_id)
        if not hrv_data.empty:
            hrv_data = extract_hrv_features(hrv_data, patient_id)
            adhd_label = patient_df.loc[patient_df["ID"] == patient_id, "ADHD"].values[0]
            hrv_data["ADHD"] = adhd_label
            all_data.append(hrv_data)

    return pd.concat(all_data, ignore_index=True)  

def extract_hrv_features(patient_df, pid):
    cfg = tsfel.get_features_by_domain("statistical")
    time_diffs = patient_df['TIME'].diff().dt.total_seconds()
    median_diff = time_diffs.median()
    fs = 1.0 / median_diff if pd.notna(median_diff) and median_diff > 0 else 1.0
    
    signal_df = pd.DataFrame(patient_df['HRV'].values)
    
    feats = tsfel.time_series_features_extractor(cfg, signal_df, fs=fs, verbose=0)
    # feats = feats.loc[:, ~feats.columns.duplicated()]
    feats['ID'] = pid
    return feats
    