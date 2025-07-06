import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight


def evaluate_from_raw_data(features_data, patient_label):
    adhd_label = LabelEncoder().fit_transform(patient_label)

    model = Sequential([
        Dense(128, activation="relu", input_shape=(features_data.shape[1],)),
        Dropout(0.5),
        Dense(64, activation="relu"),
        Dropout(0.5),
        Dense(1, activation="sigmoid")
    ])

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(patient_label), y=adhd_label)
    class_weights_dict = dict(enumerate(class_weights))

    model.fit(features_data, patient_label, epochs=15, batch_size=30, class_weight=class_weights_dict)

    return model
