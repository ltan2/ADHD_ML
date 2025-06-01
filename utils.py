import os
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

# def process_read_all_files_in_folder(folder):
#     dataframes = pd.DataFrame()
#     folderpath = os.path.join(os.getcwd(), "dataset", folder)
#     # Loop through all files in the folder
#     for filename in os.listdir(folderpath):
#         if filename.endswith('.csv'):
#             file_path = os.path.join(folderpath, filename)
#             df = pd.read_csv(file_path delimeter=";",header=0)

#             dataframes.append(df)  # Add the dataframe to the list

#     # Combine all dataframes into one (optional)
#     combined_df = pd.concat(dataframes, ignore_index=True)

#     # Print the combined dataframe (or save it)
#     print(combined_df)
