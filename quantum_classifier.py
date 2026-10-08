from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler
import numpy as np

# Example cybersecurity sample
sample = np.array([
    -0.0076786,
    -0.00491864,
    -0.71704549,
    -0.81888998
])

# Create a 4-qubit circuit
qc = QuantumCircuit(4)

# -----------------------------
# 1. Encode classical features
# -----------------------------
for i in range(4):
    qc.ry(sample[i], i)

# -----------------------------
# 2. Entangle the qubits
# -----------------------------
qc.cx(0, 1)
qc.cx(1, 2)
qc.cx(2, 3)

# -----------------------------
# 3. Add trainable parameters
# -----------------------------
weights = [0.5, 0.7, 0.3, 0.6]

for i in range(4):
    qc.ry(weights[i], i)

# -----------------------------
# 4. Measure
# -----------------------------
qc.measure_all()

print("Quantum classifier circuit:")
print(qc)

# -----------------------------
# 5. Run simulator
# -----------------------------
sampler = StatevectorSampler()

job = sampler.run([qc], shots=1000)

result = job.result()

counts = result[0].data.meas.get_counts()

print("\nMeasurement results:")
print(counts)