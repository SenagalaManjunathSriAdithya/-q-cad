import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from scipy.optimize import minimize


# ==========================================
# 1. Dataset columns
# ==========================================

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


# ==========================================
# 2. Load training dataset
# ==========================================

train_df = pd.read_csv(
    "KDDTrain+.txt",
    header=None,
    names=columns
)

train_df["label"] = train_df["attack"].apply(
    lambda x: 0 if x == "normal" else 1
)


# ==========================================
# 3. Select quantum features
# ==========================================

features = [
    "src_bytes",
    "dst_bytes",
    "count",
    "dst_host_srv_count"
]

X = train_df[features]
y = train_df["label"]


# ==========================================
# 4. Create training subset
# ==========================================

X_train, _, y_train, _ = train_test_split(
    X,
    y,
    train_size=500,
    random_state=42,
    stratify=y
)


# ==========================================
# 5. Scale training data
# ==========================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)


print("Quantum training samples:", len(X_train))


# ==========================================
# 6. Quantum prediction function
# ==========================================

def quantum_prediction(sample, weights):

    qc = QuantumCircuit(4)

    # Encode four classical features
    for i in range(4):
        qc.ry(sample[i], i)

    # Entangle qubits
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)

    # Trainable quantum layer
    for i in range(4):
        qc.ry(weights[i], i)

    # Simulate circuit
    state = Statevector.from_instruction(qc)

    probabilities = state.probabilities_dict()

    # q3 is the output qubit
    probability_attack = 0.0

    for bitstring, probability in probabilities.items():

        if bitstring[0] == "1":
            probability_attack += probability

    return probability_attack


# ==========================================
# 7. Loss function
# ==========================================

def loss_function(weights):

    total_loss = 0.0

    for sample, actual in zip(X_train, y_train):

        probability = quantum_prediction(
            sample,
            weights
        )

        probability = np.clip(
            probability,
            1e-7,
            1 - 1e-7
        )

        loss = -(
            actual * np.log(probability)
            +
            (1 - actual) * np.log(1 - probability)
        )

        total_loss += loss

    return total_loss / len(X_train)


# ==========================================
# 8. Initial weights
# ==========================================

initial_weights = np.array([
    0.5,
    0.7,
    0.3,
    0.6
])

print("\nInitial weights:")
print(initial_weights)

print("\nInitial loss:")
print(loss_function(initial_weights))


# ==========================================
# 9. Train quantum classifier
# ==========================================

print("\nTraining quantum classifier...")

result = minimize(
    loss_function,
    initial_weights,
    method="COBYLA",
    options={
        "maxiter": 100,
        "disp": True
    }
)

trained_weights = result.x


# ==========================================
# 10. Training results
# ==========================================

print("\nTraining completed!")

print("\nTrained weights:")
print(trained_weights)

print("\nFinal loss:")
print(loss_function(trained_weights))


# ==========================================
# 11. Load REAL KDDTest+ dataset
# ==========================================

print("\nLoading KDDTest+.txt...")

test_df = pd.read_csv(
    "KDDTest+.txt",
    header=None,
    names=columns
)

test_df["label"] = test_df["attack"].apply(
    lambda x: 0 if x == "normal" else 1
)


# ==========================================
# 12. Prepare test data
# ==========================================

X_test = test_df[features]
y_test = test_df["label"]

# IMPORTANT:
# Use the scaler fitted on training data.
# Do NOT fit a new scaler on test data.

X_test = scaler.transform(X_test)

print("Real test samples:", len(X_test))


# ==========================================
# 13. Quantum predictions
# ==========================================

print("\nRunning quantum predictions...")

probabilities = []

for sample in X_test:

    probability = quantum_prediction(
        sample,
        trained_weights
    )

    probabilities.append(probability)


# ==========================================
# 14. Convert probabilities to predictions
# ==========================================

predictions = [
    1 if probability >= 0.5 else 0
    for probability in probabilities
]


# ==========================================
# 15. Calculate metrics
# ==========================================

accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0
)

cm = confusion_matrix(
    y_test,
    predictions
)


# ==========================================
# 16. Display final results
# ==========================================

print("\n==========================================")
print("FINAL QUANTUM CLASSIFIER RESULTS")
print("==========================================")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nConfusion Matrix:")
print(cm)


# ==========================================
# 17. Show first 10 predictions
# ==========================================

print("\nFirst 10 KDDTest+ predictions:")

for i in range(10):

    print(
        f"Sample {i + 1}: "
        f"Probability={probabilities[i]:.4f}, "
        f"Prediction={predictions[i]}, "
        f"Actual={y_test.iloc[i]}"
    )