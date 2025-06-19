from activity_model import *
from hrv_model import *
from extract_pca import *
from p_value import *
from time_series_model import *
from utils import *
from sklearn.model_selection import train_test_split


def main():
    patient_data = read_patient_info("patient_info.csv")
    patient_ids = patient_data["ID"]

    # extract features from tsfresh
    process_activity_files(patient_ids).to_csv("dataset/full_extracted_activity_1.csv")
    process_hrv_files(patient_ids).to_csv("dataset/full_extracted_hrv.csv")

    # using PCA for feature selection
    extract_imp_features_pca("dataset/full_extracted_activity.csv", "dataset/reduced_pca_activity_1.csv")
    extract_imp_features_pca("dataset/full_extracted_hrv.csv", "dataset/reduced_hrv_activity.csv")

    # using p value for feature selection
    extract_with_pvalue("dataset/full_extracted_activity.csv", "dataset/patient_info.csv", "dataset/p_value_activity_1.csv", "ACT")
    extract_with_pvalue("dataset/full_extracted_hrv.csv", "dataset/patient_info.csv", "dataset/p_value_hrv.csv", "HRV")

    # model evaluation with raw data
    all_act_data = read_all_files_act(patient_ids)
    all_hrv_data = read_all_files_hrv(patient_ids)

    # split data to test and train
    X_train_id, X_test_id, y_train, y_test = train_test_split(patient_ids, patient_data["ADHD"], test_size=0.2, random_state=42, stratify=patient_data["ADHD"])
    
    # act data
    X_train = all_act_data[all_act_data["ID"].isin(X_train_id)]
    X_test = all_act_data[all_act_data["ID"].isin(X_test_id)]
    X_train_m = normalized_timestamps(X_train)
    X_test_m = normalized_timestamps(X_test)

    tst_model_fit = tst_model(X_train_m, y_train)
    
    # Evaluate the model on test data
    test_loss, test_accuracy = tst_model_fit.evaluate(X_test_m, y_test)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")




if __name__ == "__main__":
    main()
