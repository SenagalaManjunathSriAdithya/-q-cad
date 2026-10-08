import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

import joblib

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize


# ============================================================
# SETTINGS
# ============================================================

MODEL_DIR = "model"

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# DATASET COLUMNS
# ============================================================

columns = [
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
    "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count",
    "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "attack", "difficulty"
]


# ============================================================
# LOAD DATA
# ============================================================

print("Loading dataset...")

train = pd.read_csv(
    "KDDTrain+.txt",
    header=None,
    names=columns
)

train["label"] = train["attack"].apply(
    lambda x: 0 if x == "normal" else 1
)

print("Dataset loaded:", train.shape)


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

X = train.drop(
    columns=["attack", "difficulty", "label"]
)

y = train["label"]

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))


# ============================================================
# CLASSICAL MODEL
# ============================================================

print("\nTraining Classical Random Forest...")

categorical_features = [
    "protocol_type",
    "service",
    "flag"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)

classical_model = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "model",
        RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
    )
])

classical_model.fit(
    X_train,
    y_train
)

print("Classical model trained.")


# ============================================================
# SAVE CLASSICAL MODEL
# ============================================================

joblib.dump(
    classical_model,
    os.path.join(
        MODEL_DIR,
        "classical_model.pkl"
    )
)

print("Saved: model/classical_model.pkl")


# ============================================================
# QUANTUM FEATURES
# ============================================================

quantum_features = [
    "src_bytes",
    "dst_bytes",
    "count",
    "dst_host_srv_count"
]


# ============================================================
# QUANTUM PREDICTION
# ============================================================

def quantum_prediction(sample, weights):

    qc = QuantumCircuit(4)

    for i in range(4):
        qc.ry(sample[i], i)

    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)

    for i in range(4):
        qc.ry(weights[i], i)

    state = Statevector.from_instruction(qc)

    probabilities = state.probabilities_dict()

    probability_attack = 0.0

    for bitstring, probability in probabilities.items():

        if bitstring[0] == "1":
            probability_attack += probability

    return probability_attack


# ============================================================
# QUANTUM LOSS
# ============================================================

def quantum_loss(weights, X_data, y_data):

    predictions = []

    for sample in X_data:

        probability = quantum_prediction(
            sample,
            weights
        )

        predictions.append(probability)

    predictions = np.array(predictions)

    predictions = np.clip(
        predictions,
        1e-7,
        1 - 1e-7
    )

    loss = -np.mean(
        y_data * np.log(predictions)
        +
        (1 - y_data) * np.log(1 - predictions)
    )

    return loss


# ============================================================
# PREPARE QUANTUM TRAINING DATA
# ============================================================

print("\nPreparing Quantum Model...")

X_quantum = X_train[
    quantum_features
]

y_quantum = y_train.values


# Use 500 samples, same experimental setup
X_q_train, _, y_q_train, _ = train_test_split(
    X_quantum,
    y_quantum,
    train_size=500,
    random_state=42,
    stratify=y_quantum
)


# ============================================================
# QUANTUM SCALER
# ============================================================

scaler = StandardScaler()

X_q_train_scaled = scaler.fit_transform(
    X_q_train
)

# Convert scaled values to quantum rotation angles
X_q_train_scaled = (
    np.tanh(X_q_train_scaled)
    * (np.pi / 2)
)


# ============================================================
# TRAIN QUANTUM MODEL
# ============================================================

initial_weights = np.array([
    0.5,
    0.7,
    0.3,
    0.6
])

print("\nTraining Quantum Model...")

result = minimize(
    quantum_loss,
    initial_weights,
    args=(
        X_q_train_scaled,
        y_q_train
    ),
    method="COBYLA",
    options={
        "maxiter": 100
    }
)

quantum_weights = result.x

print("Quantum model trained.")

print("Quantum weights:")
print(quantum_weights)

print("Quantum loss:")
print(result.fun)


# ============================================================
# SAVE QUANTUM SCALER
# ============================================================

joblib.dump(
    scaler,
    os.path.join(
        MODEL_DIR,
        "quantum_scaler.pkl"
    )
)

print("Saved: model/quantum_scaler.pkl")


# ============================================================
# SAVE QUANTUM WEIGHTS
# ============================================================

np.save(
    os.path.join(
        MODEL_DIR,
        "quantum_weights.npy"
    ),
    quantum_weights
)

print("Saved: model/quantum_weights.npy")


# ============================================================
# HYBRID CONFIGURATION
# ============================================================

hybrid_config = {
    "classical_weight": 0.95,
    "quantum_weight": 0.05,
    "threshold": 0.45,
    "quantum_features": quantum_features
}


# ============================================================
# SAVE HYBRID CONFIGURATION
# ============================================================

joblib.dump(
    hybrid_config,
    os.path.join(
        MODEL_DIR,
        "hybrid_config.pkl"
    )
)

print("Saved: model/hybrid_config.pkl")


# ============================================================
# COMPLETE
# ============================================================

print("\n========================================")
print("ALL MODELS SAVED SUCCESSFULLY")
print("========================================")

print("\nCreated files:")

print("1. model/classical_model.pkl")
print("2. model/quantum_scaler.pkl")
print("3. model/quantum_weights.npy")
print("4. model/hybrid_config.pkl")