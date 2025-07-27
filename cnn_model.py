import numpy as np
import os
import random
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from scikeras.wrappers import KerasClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# SEED = 12
# os.environ['PYTHONHASHSEED'] = str(SEED)
# random.seed(SEED)
# np.random.seed(SEED)
# tf.random.set_seed(SEED)


def evaluate_from_raw_data(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation="relu"),
        Dropout(0.2),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(1, activation="sigmoid")
    ])

    model.compile(optimizer=Adam(learning_rate=0.005), loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_run_eval_model(X, y):
    # Preprocessing
    y_encoded = LabelEncoder().fit_transform(y)
    X_scaled = StandardScaler().fit_transform(X)
    input_dim = X_scaled.shape[1]

    # Compute class weights
    class_weights_array = compute_class_weight(class_weight='balanced', classes=np.unique(y_encoded), y=y_encoded)
    class_weights_dict = dict(enumerate(class_weights_array))

    # Wrap model
    clf = KerasClassifier(
        model=lambda: evaluate_from_raw_data(input_dim),
        epochs=10,
        batch_size=40,
        verbose=0,
        class_weight=class_weights_dict,
    )

    # 10-fold stratified cross-validation
    cv = StratifiedKFold(n_splits=10, shuffle=True)

    results = cross_validate(clf, X_scaled, y_encoded, cv=cv,
                             scoring=['accuracy', 'f1'],
                             return_train_score=False)

    print("Cross-validation results (10-fold):")
    print("Accuracy scores:", results['test_accuracy'])
    print("F1 scores:", results['test_f1'])
    print(f"Mean Accuracy: {np.mean(results['test_accuracy']):.4f}")
    print(f"Mean F1 Score: {np.mean(results['test_f1']):.4f}")

