import os
from datetime import datetime
import csv
import pandas as pd
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
                    timestamp = datetime.strptime(line[0], "%m-%d-%Y %H:%M").timestamp()
                    activity = int(line[1].split(" ")[0])
                    data.append([timestamp, activity])
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        return pd.DataFrame()
    return pd.DataFrame(data, columns=["TIME", "ACT"]).assign(ID=patient_id)

def process_activity_files(patient_ids):
    """
    Process all patient activity files.
    
    Args:
        patient_ids: List of all patients

    Returns:
        pandas dataframe of extracted features from all patient
    """
    patients_df = read_all_files(patient_ids)

    if patients_df:
        extracted_features = extract_features(patients_dfs, column_id="ID", column_value="ACT", column_sort="TIME", n_jobs=0, show_warnings=False)
        return extracted_features
    
    return pd.DataFrame()

def read_all_files(patient_ids):
    """
    Process all patient activity files.
    
    Args:
        patient_ids: List of all patients

    Returns:
        pandas dataframe of from all patient
    """
    all_data = []
    for patient_id in patient_ids:
        activity_data = read_activity_file(patient_id)
        if not activity_data.empty:
            all_data.append(activity_data)

    return pd.concat(all_data, ignore_index=True)  
