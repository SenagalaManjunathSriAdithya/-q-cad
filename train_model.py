import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

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

# Load training data
train = pd.read_csv("KDDTrain+.txt", header=None, names=columns)

# Convert attack type to binary label
train["label"] = train["attack"].apply(
    lambda x: 0 if x == "normal" else 1
)

# Remove columns that are not useful for prediction
X = train.drop(columns=["attack", "difficulty", "label"])
y = train["label"]

# Find categorical columns
categorical_features = ["protocol_type", "service", "flag"]

# Everything else is numerical
numeric_features = [
    col for col in X.columns
    if col not in categorical_features
]

# Convert text categories into numbers
preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"),
         categorical_features)
    ],
    remainder="passthrough"
)

# Random Forest classifier
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

# Combine preprocessing + model
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

print("Training model...")

pipeline.fit(X, y)

print("Model trained successfully!")

# Load test data
test = pd.read_csv("KDDTest+.txt", header=None, names=columns)

# Convert attack type to binary label
test["label"] = test["attack"].apply(
    lambda x: 0 if x == "normal" else 1
)

# Prepare test features and labels
X_test = test.drop(columns=["attack", "difficulty", "label"])
y_test = test["label"]

# Make predictions
print("\nTesting model...")

predictions = pipeline.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:", accuracy)

# Detailed performance
print("\nClassification Report:")
print(classification_report(y_test, predictions))