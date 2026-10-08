import pandas as pd
from sklearn.preprocessing import StandardScaler

# Column names
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

# Load dataset
df = pd.read_csv(
    "KDDTrain+.txt",
    header=None,
    names=columns
)

# Convert attack type into binary label
df["label"] = df["attack"].apply(
    lambda x: 0 if x == "normal" else 1
)

# Select four features for the quantum circuit
features = [
    "src_bytes",
    "dst_bytes",
    "count",
    "dst_host_srv_count"
]

X = df[features]
y = df["label"]

# Scale the values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Quantum features:")
print(features)

print("\nOriginal shape:", X.shape)
print("Scaled shape:", X_scaled.shape)

print("\nFirst 5 scaled samples:")
print(X_scaled[:5])

print("\nLabels:")
print(y.head())