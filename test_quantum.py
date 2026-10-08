from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler

# Create a quantum circuit
qc = QuantumCircuit(1)

# Put the qubit into superposition
qc.h(0)

# Measure the qubit
qc.measure_all()

# Create simulator
sampler = StatevectorSampler()

# Run the circuit 1000 times
job = sampler.run([qc], shots=1000)

# Get results
result = job.result()

# Get measurement counts
counts = result[0].data.meas.get_counts()

print("Measurement results:")
print(counts)