import os
from datetime import datetime
import csv
import pandas as pd
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
                    # discarding seconds
                    timestamp = f"{timestamp.hour:02d}:{timestamp.minute:02d}"
                    activity = float(line[1].split(" ")[0])
                    data.append([timestamp, activity])
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        return pd.DataFrame()
    return pd.DataFrame(data, columns=["TIME", "HRV"]).assign(ID=patient_id)

def process_hrv_files(patient_ids):
    patients_df = read_all_files_hrv(patient_ids)

    if patients_df:  
        extracted_features = extract_features(patients_df, column_id="ID", column_value="HRV", column_sort="TIME", n_jobs=0, show_warnings=False)
        return extracted_features
    
    return pd.DataFrame()

def read_all_files_hrv(patient_ids):
    """
    Process all patient hrv files.
    
    Args:
        patient_ids: List of all patients

    Returns:
        pandas dataframe of from all patient
    """
    all_data = []
    for patient_id in patient_ids:
        activity_data = read_hrv_file(patient_id)
        if not activity_data.empty:
            all_data.append(activity_data)

    return pd.concat(all_data, ignore_index=True)  
