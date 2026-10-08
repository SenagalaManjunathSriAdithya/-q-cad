from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

import numpy as np
import pandas as pd
import joblib

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


# ============================================================
# FLASK APP
# ============================================================

app = Flask(
    __name__,
    static_folder="frontend",
    static_url_path=""
)

CORS(app)


# ============================================================
# LOAD SAVED MODELS
# ============================================================

print("Loading Q-CAD models...")

classical_model = joblib.load(
    "model/classical_model.pkl"
)

quantum_scaler = joblib.load(
    "model/quantum_scaler.pkl"
)

quantum_weights = np.load(
    "model/quantum_weights.npy"
)

hybrid_config = joblib.load(
    "model/hybrid_config.pkl"
)

CLASSICAL_WEIGHT = hybrid_config["classical_weight"]
QUANTUM_WEIGHT = hybrid_config["quantum_weight"]
THRESHOLD = hybrid_config["threshold"]

QUANTUM_FEATURES = hybrid_config["quantum_features"]


print("Models loaded successfully.")

print("Classical weight:", CLASSICAL_WEIGHT)
print("Quantum weight:", QUANTUM_WEIGHT)
print("Threshold:", THRESHOLD)


# ============================================================
# QUANTUM PREDICTION
# ============================================================

def quantum_prediction(sample):

    qc = QuantumCircuit(4)

    for i in range(4):
        qc.ry(sample[i], i)

    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)

    for i in range(4):
        qc.ry(quantum_weights[i], i)

    state = Statevector.from_instruction(qc)

    probabilities = state.probabilities_dict()

    probability_attack = 0.0

    for bitstring, probability in probabilities.items():

        if bitstring[0] == "1":
            probability_attack += probability

    return float(probability_attack)


# ============================================================
# QUANTUM PROBABILITY
# ============================================================

def get_quantum_probability(data):

    values = np.array([
        data["src_bytes"],
        data["dst_bytes"],
        data["count"],
        data["dst_host_srv_count"]
    ], dtype=float)

    values = values.reshape(1, -1)

    scaled = quantum_scaler.transform(values)

    scaled = np.tanh(scaled) * (np.pi / 2)

    return quantum_prediction(
        scaled[0]
    )


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/", methods=["GET"])
def home():

    return send_from_directory(
        "frontend",
        "index.html"
    )


# ============================================================
# PREDICTION API
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json()

        if not data:

            return jsonify({
                "error": "No JSON data received"
            }), 400


        # ----------------------------------------------------
        # Required input fields
        # ----------------------------------------------------

        required_fields = [
            "src_bytes",
            "dst_bytes",
            "count",
            "dst_host_srv_count"
        ]

        for field in required_fields:

            if field not in data:

                return jsonify({
                    "error": f"Missing field: {field}"
                }), 400


        # ----------------------------------------------------
        # Create classical input
        # ----------------------------------------------------

        classical_data = {

            "duration": 0,

            "protocol_type": "tcp",

            "service": "http",

            "flag": "SF",

            "src_bytes": float(
                data["src_bytes"]
            ),

            "dst_bytes": float(
                data["dst_bytes"]
            ),

            "land": 0,

            "wrong_fragment": 0,

            "urgent": 0,

            "hot": 0,

            "num_failed_logins": 0,

            "logged_in": 1,

            "num_compromised": 0,

            "root_shell": 0,

            "su_attempted": 0,

            "num_root": 0,

            "num_file_creations": 0,

            "num_shells": 0,

            "num_access_files": 0,

            "num_outbound_cmds": 0,

            "is_host_login": 0,

            "is_guest_login": 0,

            "count": float(
                data["count"]
            ),

            "srv_count": 0,

            "serror_rate": 0,

            "srv_serror_rate": 0,

            "rerror_rate": 0,

            "srv_rerror_rate": 0,

            "same_srv_rate": 0,

            "diff_srv_rate": 0,

            "srv_diff_host_rate": 0,

            "dst_host_count": 0,

            "dst_host_srv_count": float(
                data["dst_host_srv_count"]
            ),

            "dst_host_same_srv_rate": 0,

            "dst_host_diff_srv_rate": 0,

            "dst_host_same_src_port_rate": 0,

            "dst_host_srv_diff_host_rate": 0,

            "dst_host_serror_rate": 0,

            "dst_host_srv_serror_rate": 0,

            "dst_host_rerror_rate": 0,

            "dst_host_srv_rerror_rate": 0
        }


        classical_df = pd.DataFrame([
            classical_data
        ])


        # ----------------------------------------------------
        # Classical probability
        # ----------------------------------------------------

        classical_probability = float(

            classical_model.predict_proba(
                classical_df
            )[0][1]

        )


        # ----------------------------------------------------
        # Quantum probability
        # ----------------------------------------------------

        quantum_probability = get_quantum_probability(
            data
        )


        # ----------------------------------------------------
        # Hybrid score
        # ----------------------------------------------------

        hybrid_probability = (

            CLASSICAL_WEIGHT *
            classical_probability

            +

            QUANTUM_WEIGHT *
            quantum_probability

        )


        # ----------------------------------------------------
        # Final prediction
        # ----------------------------------------------------

        if hybrid_probability >= THRESHOLD:

            prediction = "ANOMALY"

        else:

            prediction = "NORMAL"


        # ----------------------------------------------------
        # Response
        # ----------------------------------------------------

        return jsonify({

            "prediction": prediction,

            "classical_probability": round(
                classical_probability,
                4
            ),

            "quantum_probability": round(
                quantum_probability,
                4
            ),

            "hybrid_probability": round(
                hybrid_probability,
                4
            ),

            "threshold": THRESHOLD

        })


    except Exception as error:

        return jsonify({

            "error": str(error)

        }), 500


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print("\n========================================")
    print("Q-CAD BACKEND")
    print("========================================")
    print("Server starting...")
    print("========================================\n")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )