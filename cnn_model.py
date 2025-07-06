import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report


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

def train_run_eval_model(X, y):
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