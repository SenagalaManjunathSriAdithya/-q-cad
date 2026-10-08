import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
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


# ============================================================
# 1. DATASET
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

print("Loading dataset...")

train = pd.read_csv(
    "KDDTrain+.txt",
    header=None,
    names=columns
)

test = pd.read_csv(
    "KDDTest+.txt",
    header=None,
    names=columns
)

train["label"] = train["attack"].apply(
    lambda x: 0 if x == "normal" else 1
)

test["label"] = test["attack"].apply(
    lambda x: 0 if x == "normal" else 1
)

print("Training dataset:", train.shape)
print("Test dataset:", test.shape)


# ============================================================
# 2. CLASSICAL MODEL
# ============================================================

X = train.drop(columns=["attack", "difficulty", "label"])
y = train["label"]

X_test = test.drop(columns=["attack", "difficulty", "label"])
y_test = test["label"]

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


# ============================================================
# 3. TRAIN / VALIDATION SPLIT
# ============================================================

print("\nCreating validation split...")

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
# 4. TRAIN CLASSICAL MODEL
# ============================================================

print("\nTraining Classical Random Forest...")

classical_model.fit(X_train, y_train)

print("Classical model trained.")


# ============================================================
# 5. QUANTUM FEATURES
# ============================================================

quantum_features = [
    "src_bytes",
    "dst_bytes",
    "count",
    "dst_host_srv_count"
]


# ============================================================
# 6. QUANTUM PREDICTION
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
# 7. QUANTUM LOSS
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
# 8. TRAIN QUANTUM MODEL
# ============================================================

print("\nPreparing quantum training data...")

X_quantum_train = X_train[quantum_features]
y_quantum_train = y_train.values

# Use 500 stratified samples to keep simulation practical
X_q_train, _, y_q_train, _ = train_test_split(
    X_quantum_train,
    y_quantum_train,
    train_size=500,
    random_state=42,
    stratify=y_quantum_train
)

scaler = StandardScaler()

X_q_train_scaled = scaler.fit_transform(
    X_q_train
)

print("Quantum training samples:", len(X_q_train_scaled))


# Convert values to angles
X_q_train_scaled = np.tanh(
    X_q_train_scaled
) * (np.pi / 2)


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

print("Quantum training completed.")

print("Quantum weights:")
print(quantum_weights)

print("Final quantum loss:")
print(result.fun)


# ============================================================
# 9. QUANTUM PROBABILITY FUNCTION
# ============================================================

def get_quantum_probabilities(dataframe):

    quantum_data = dataframe[
        quantum_features
    ]

    scaled = scaler.transform(
        quantum_data
    )

    scaled = np.tanh(
        scaled
    ) * (np.pi / 2)

    probabilities = []

    for sample in scaled:

        probability = quantum_prediction(
            sample,
            quantum_weights
        )

        probabilities.append(probability)

    return np.array(probabilities)


# ============================================================
# 10. VALIDATION PROBABILITIES
# ============================================================

print("\nGenerating validation probabilities...")

classical_val_prob = classical_model.predict_proba(
    X_val
)[:, 1]

quantum_val_prob = get_quantum_probabilities(
    X_val
)

print("Validation probabilities generated.")


# ============================================================
# 11. FIND HYBRID PARAMETERS
# ============================================================

print("\nSearching hybrid parameters...")

best_f1 = -1
best_alpha = None
best_threshold = None

# IMPORTANT:
# We require a real quantum contribution.
# Alpha cannot be 1.0.

for alpha in np.arange(
    0.1,
    1.0,
    0.05
):

    hybrid_val_prob = (
        alpha * classical_val_prob
        +
        (1 - alpha) * quantum_val_prob
    )

    for threshold in np.arange(
        0.30,
        0.71,
        0.05
    ):

        predictions = (
            hybrid_val_prob >= threshold
        ).astype(int)

        score = f1_score(
            y_val,
            predictions
        )

        if score > best_f1:

            best_f1 = score
            best_alpha = alpha
            best_threshold = threshold


print("\nBest Hybrid Parameters")

print(
    "Classical weight:",
    round(best_alpha, 2)
)

print(
    "Quantum weight:",
    round(1 - best_alpha, 2)
)

print(
    "Threshold:",
    round(best_threshold, 2)
)

print(
    "Validation F1:",
    round(best_f1, 4)
)


# ============================================================
# 12. FINAL TEST PREDICTIONS
# ============================================================

print("\nGenerating final test predictions...")

classical_test_prob = classical_model.predict_proba(
    X_test
)[:, 1]

quantum_test_prob = get_quantum_probabilities(
    X_test
)

hybrid_test_prob = (
    best_alpha * classical_test_prob
    +
    (1 - best_alpha) * quantum_test_prob
)

classical_pred = (
    classical_test_prob >= 0.5
).astype(int)

quantum_pred = (
    quantum_test_prob >= 0.5
).astype(int)

hybrid_pred = (
    hybrid_test_prob >= best_threshold
).astype(int)


# ============================================================
# 13. EVALUATION FUNCTION
# ============================================================

def evaluate_model(name, predictions):

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions
    )

    recall = recall_score(
        y_test,
        predictions
    )

    f1 = f1_score(
        y_test,
        predictions
    )

    matrix = confusion_matrix(
        y_test,
        predictions
    )

    print("\n" + "=" * 50)

    print(name)

    print("=" * 50)

    print(
        "Accuracy :",
        round(accuracy, 4)
    )

    print(
        "Precision:",
        round(precision, 4)
    )

    print(
        "Recall   :",
        round(recall, 4)
    )

    print(
        "F1 Score :",
        round(f1, 4)
    )

    print("\nConfusion Matrix:")

    print(matrix)


# ============================================================
# 14. FINAL COMPARISON
# ============================================================

evaluate_model(
    "CLASSICAL RANDOM FOREST",
    classical_pred
)

evaluate_model(
    "QUANTUM MODEL",
    quantum_pred
)

evaluate_model(
    "HYBRID MODEL",
    hybrid_pred
)


# ============================================================
# 15. SAMPLE HYBRID RESULTS
# ============================================================

print("\n" + "=" * 60)

print("FIRST 10 HYBRID PREDICTIONS")

print("=" * 60)

for i in range(10):

    print(
        f"{i + 1}. "
        f"Classical={classical_test_prob[i]:.4f} "
        f"Quantum={quantum_test_prob[i]:.4f} "
        f"Hybrid={hybrid_test_prob[i]:.4f} "
        f"Pred={hybrid_pred[i]} "
        f"Actual={y_test.iloc[i]}"
    )