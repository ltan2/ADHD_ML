from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from sklearn.preprocessing import LabelEncoder


def evaluate_from_raw_data(features_data, patient_label):
    adhd_label = LabelEncoder().fit_transform(patient_label)

    model = Sequential([
        Conv1D(64, kernel_size=3, activation="relu", input_shape=(features_data.shape[1], features_data.shape[2])),
        MaxPooling1D(pool_size=2),
        Conv1D(128, kernel_size=3, activation="relu"),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(1, activation="sigmoid")
    ])

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    model.fit(features_data, adhd_label, epochs=10, batch_size=16, validation_split=0.2)
