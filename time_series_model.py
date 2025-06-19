import numpy as np
import pandas as pd
from tensorflow.keras.layers import Input, Dense, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D
from tensorflow.keras.models import Model

def tst_model(all_patient_data, all_patient_label):
    input_layer = Input(shape=(all_patient_data.shape[1], 1)) 

    attention = MultiHeadAttention(num_heads=4, key_dim=64)(input_layer, input_layer)
    normalized = LayerNormalization()(attention)

    dense = Dense(128, activation="relu")(normalized)
    pooled = GlobalAveragePooling1D()(dense)
    output = Dense(1, activation="sigmoid")(pooled)

    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    model.fit(all_patient_data, all_patient_label, epochs=10, batch_size=16, validation_split=0.2)

    return model

def normalized_timestamps(all_patient):
    group_patients = all_patient.groupby("ID")

    patient_aligned_data = {}
    common_timestamps = set.intersection(*all_patient.groupby("ID")["TIME"].apply(set))
    common_timestamps = set(map(str, common_timestamps))


    for pid, group in group_patients:
        group = group.drop_duplicates(subset=["TIME"])
        patient_series = group.set_index("TIME")["ACT"]
        patient_series.index = patient_series.index.astype(str)
        # drop if not match
        aligned_series = patient_series[patient_series.index.isin(common_timestamps)]
        # pad
        aligned_series = aligned_series.reindex(common_timestamps, fill_value=0)
        patient_aligned_data[pid] = aligned_series.values

    sorted_patient_ids = sorted(patient_aligned_data.keys())
    X = np.array([patient_aligned_data[id] for id in sorted_patient_ids]).reshape(len(sorted_patient_ids), len(common_timestamps), 1)
    print("Shape of X:", X.shape)
    return X


    
