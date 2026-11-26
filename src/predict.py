import json
import joblib
import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path


class EnergyNN_v3(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(11, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

# Load configuration
with open(MODEL_DIR / "model_config_v3.json") as f:
    config = json.load(f)

# Load scaler
scaler = joblib.load(MODEL_DIR / "energy_scaler_v3.pkl")

# Initialize model architecture
model = EnergyNN_v3()

# Load trained weights
model.load_state_dict(
    torch.load(
        MODEL_DIR / "energy_nn_v3.pth",
        map_location="cpu",
        weights_only=True
    )
)

# Set model to evaluation mode
model.eval()

print("Model loaded successfully!")

def predict(input_data: pd.DataFrame):
    # Check required features
    required_features = config["features"]

    missing_features = [
        feature for feature in required_features
        if feature not in input_data.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing required features: {missing_features}"
        )

    # Select features in the correct order
    X = input_data[required_features]

    # Apply the same scaler used during training
    X_scaled = scaler.transform(X)

    # Convert to PyTorch tensor
    X_tensor = torch.tensor(
        X_scaled,
        dtype=torch.float32
    )

    # Generate predictions
    with torch.no_grad():
        predictions = model(X_tensor).numpy().flatten()

    return predictions

if __name__ == "__main__":
    import pandas as pd

    # Load processed dataset
    data = pd.read_csv(BASE_DIR / "data" / "processed" / "energy_features.csv")

    # Select one row containing all required features
    sample = data[config["features"]].iloc[[100]]

    # Generate prediction
    result = predict(sample)

    print("Predicted energy consumption in 6 hours:", result[0], "kWh")