from activity_model import *
from cnn_model import *
from hrv_model import *
from extract_pca import *
from p_value import *
from time_series_model import *
from utils import *

def main():
    # extract features from tsfresh
    # hrv_df = read_all_files_hrv("patient_info.csv")
    # hrv_df["TIME"] = hrv_df["TIME"].astype("int64")
    # hrv_X_df = tsfresh_pvalue(hrv_df.drop(columns=["ADHD"]), hrv_df["ADHD"])
    # hrv_df.to_csv("dataset/hrv_tsfel_features.csv")   

    act_df = read_all_files_act("patient_info.csv")
    # act_df["TIME"] = act_df["TIME"].astype("int64")
    # # hrv_df["TIME"] = hrv_df["TIME"].astype("int64")
    # # hrv_X_df = tsfresh_pvalue(hrv_df.drop(columns=["ADHD"]), hrv_df["ADHD"])
    act_df.to_csv("dataset/act_tsfel_features.csv")   

    # # # using PCA for feature selection
    # # extract_imp_features_pca("dataset/full_extracted_activity.csv", "dataset/reduced_pca_activity_1.csv")
    # # extract_imp_features_pca("dataset/full_extracted_hrv.csv", "dataset/reduced_hrv_activity.csv")

    # # # using p value for feature selection
    # # extract_with_pvalue("dataset/full_extracted_activity.csv", "dataset/patient_info.csv", "dataset/p_value_activity_1.csv", "ACT")
    # # extract_with_pvalue("dataset/full_extracted_hrv.csv", "dataset/patient_info.csv", "dataset/p_value_hrv.csv", "HRV")

    # # # model evaluation with raw data
    # # all_act_data = read_all_files_act(patient_ids)
    # # all_hrv_data = read_all_files_hrv(patient_ids)

    # # split data to test and train
    # print("##################################### HRV MODEL #######################################3")
    # hrv_data = pd.read_csv("dataset/hrv_tsfel_features.csv", header = 0)
    # X = hrv_data.drop(columns=["ADHD", "ID"])
    # y = hrv_data["ADHD"]
    # train_run_eval_model(X,y)

    # run X model to p_value extraction
    # X = extract_with_pvalue(X,y)

    # train_run_eval_model(X,y)

    print("##################################### ACT MODEL #######################################3")

    act_data = pd.read_csv("dataset/act_tsfel_features.csv", header = 0)
    X = act_data.drop(columns=["ADHD", "ID"])
    y = act_data["ADHD"]

    # run X model to p_value extraction
    X = extract_with_pvalue(X,y)

    train_run_eval_model(X,y)

    
    # X_train_id, X_test_id, y_train, y_test = train_test_split(patient_ids, patient_data["ADHD"], test_size=0.2, random_state=42, stratify=patient_data["ADHD"])
    
    # # act data
    # X_train = all_act_data[all_act_data["ID"].isin(X_train_id)]
    # X_test = all_act_data[all_act_data["ID"].isin(X_test_id)]
    # X_train_m = normalized_timestamps(X_train)
    # X_test_m = normalized_timestamps(X_test)

    # tst_model_fit = tst_model(X_train_m, y_train)
    
    # # Evaluate the model on test data
    # test_loss, test_accuracy = tst_model_fit.evaluate(X_test_m, y_test)
    # print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")




if __name__ == "__main__":
    main()
