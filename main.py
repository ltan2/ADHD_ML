from activity_model import *
from hrv_model import *
from extract_pca import *
from p_value import *
from utils import *

def main():
    patient_data = read_patient_info("patient_info.csv")
    patient_ids = patient_data["ID"]

    # extract features from tsfresh
    process_activity_files(patient_ids).to_csv("dataset/full_extracted_activity.csv")
    process_hrv_files(patient_ids).to_csv("dataset/full_extracted_hrv.csv")

    # using PCA for feature selection
    extract_imp_features_pca("dataset/full_extracted_activity.csv", "dataset/reduced_pca_activity.csv")

    # using p value for feature selection
    extract_with_pvalue("dataset/full_extracted_activity.csv", "dataset/patient_info.csv", "dataset/p_value_activity.csv", "ACT")

if __name__ == "__main__":
    main()
