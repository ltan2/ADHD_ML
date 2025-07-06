from activity_model import *
from cnn_model import *
from hrv_model import *
from extract_pca import *
from p_value import *
from time_series_model import *
from utils import *
from sklearn.model_selection import train_test_split
from tsfresh_pvalue import tsfresh_pvalue
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

def main():
    # extract features from tsfresh
    # hrv_df = read_all_files_hrv("patient_info.csv")
    # # hrv_df["TIME"] = hrv_df["TIME"].astype("int64")
    # # hrv_X_df = tsfresh_pvalue(hrv_df.drop(columns=["ADHD"]), hrv_df["ADHD"])
    # hrv_df.to_csv("dataset/hrv_tsfresh_pvalue_features.csv")    

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
    hrv_data = pd.read_csv("dataset/full_extracted_tsfresh.csv", header = 0)
    print(hrv_data.head())
    X = hrv_data.drop(columns=["ADHD", "ID"])
    y = hrv_data["ADHD"]
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = evaluate_from_raw_data(X_train, y_train)
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    # Step 6: Evaluate
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
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
