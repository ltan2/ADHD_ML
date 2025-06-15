import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def extract_imp_features_pca(filename, output_file):
    features = pd.read_csv(filename)
    features.fillna(0, inplace=True)
    features.replace([np.inf, -np.inf], 0, inplace=True)

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Apply PCA
    pca = PCA(n_components=0.95)  # 95% variance
    reduced_features = pca.fit_transform(scaled_features)

    reduced_df = pd.DataFrame(reduced_features)
    reduced_df.to_csv(output_file, index=False)

extract_imp_features_pca("dataset/full_extracted_activity.csv", "dataset/reduced_pca_activity.csv")