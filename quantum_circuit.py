from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler
import numpy as np

# Example scaled cybersecurity data
sample = np.array([
    -0.0076786,
    -0.00491864,
    -0.71704549,
    -0.81888998
])

# Create a 4-qubit circuit
qc = QuantumCircuit(4)

# Encode the four classical features
for i in range(4):
    qc.ry(sample[i], i)

# Entangle the qubits
qc.cx(0, 1)
qc.cx(1, 2)
qc.cx(2, 3)

# Measure all qubits
qc.measure_all()

print("Quantum circuit:")
print(qc)

# Create Qiskit simulator
sampler = StatevectorSampler()

# Run the circuit
job = sampler.run([qc], shots=1000)

# Get results
result = job.result()

# Get measurement counts
counts = result[0].data.meas.get_counts()

print("\nMeasurement results:")
print(counts)
