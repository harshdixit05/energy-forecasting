# ⚡ Energy Consumption Forecasting

An end-to-end machine learning project for forecasting industrial energy consumption using **PyTorch, FastAPI, and Streamlit**.

## Overview

This project explores time-series forecasting using historical energy consumption data from a steel industry facility.

The objective is to build a machine learning pipeline that processes historical consumption data, engineers time-based features, trains a neural network, and serves predictions through a REST API with an interactive dashboard.

The current model predicts energy consumption **6 hours ahead**.

## Tech 

- **Language:** Python
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Plotly
- **Machine Learning:** PyTorch, Scikit-learn
- **Backend:** FastAPI
- **Frontend:** Streamlit
- **Model Persistence:** Joblib, PyTorch
- **Environment & Dependencies:** uv

## Dataset

The project uses the Steel Industry Energy Consumption dataset.

- Period: January–December 2018
- Original observations: 35,040
- Sampling frequency: 15 minutes
- Target variable: `Usage_kWh`

The dataset includes energy consumption, reactive power, power factor, CO₂ emissions, time information, and load type.

*Note: The dataset is not included in this repository. Place the original CSV inside `data/raw/` before running the notebooks and data-processing workflow.*

## Feature Engineering

Time-based and historical features are used to capture consumption patterns.

Some engineered features include:

- Hour, day, and month
- Day of the week
- Weekend indicator
- Lag features: 1, 4, 96, and 672 intervals
- Rolling averages: 4 and 96 intervals

The target variable is created using a 24-step shift, corresponding to a 6-hour forecasting horizon at 15-minute intervals.

## Model Architecture

A feedforward neural network built using PyTorch.

**Architecture:**

`11 input features → 128 → 64 → 32 → 1 output`

- ReLU activation
- Dropout regularization (0.2)
- Adam optimizer
- Mean Squared Error loss

The dataset is split chronologically into training, validation, and testing sets to respect the time-series structure.

### Model Performance — V3

| Metric | Validation | Test |
|---|---:|---:|
| MAE | 11.74 | 11.65 |
| RMSE | 20.38 | 19.91 |

Metrics are expressed in the target variable's units (kWh).

## Application Architecture

The trained model is integrated into a simple prediction application.

1. **Streamlit frontend:** Displays historical consumption and allows users to request predictions.
2. **FastAPI backend:** Receives feature inputs through a REST API.
3. **Prediction module:** Loads the trained model and scaler, transforms inputs, and generates predictions.
4. **PyTorch model:** Produces the 6-hour-ahead energy consumption estimate.

## Project Structure

```text
energy-forecasting/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── energy_nn_v3.pth
│   ├── energy_scaler_v3.pkl
│   └── model_config_v3.json
│
├── src/
│   ├── 01_data_exploration.ipynb
│   ├── 03_neural_networks.ipynb
│   ├── predict.py
│   ├── api.py
│   └── app.py
│
├── .gitignore
├── pyproject.toml
├── uv.lock
└── README.md
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/harshdixit05/energy-forecasting.git
cd energy-forecasting
```

### 2. Install dependencies

Install [uv](https://docs.astral.sh/uv/) if you haven't already.

```bash
uv sync
```

### 3. Add the dataset

Place the original dataset at:

```text
data/raw/Steel_industry_data.csv
```

Run the data preparation workflow to generate the processed dataset required by the dashboard:

```text
data/processed/energy_features.csv
```

### 4. Start the FastAPI backend

```bash
uv run uvicorn src.api:app --reload
```

API documentation:

http://127.0.0.1:8000/docs

### 5. Start the Streamlit dashboard

Open a second terminal:

```bash
uv run streamlit run src/app.py
```

Dashboard:

http://localhost:8501

## Current Scope

- Historical energy consumption analysis
- Time-series feature engineering
- Neural network model training
- Model saving and loading
- REST API for predictions
- Interactive Streamlit dashboard

## Future Improvements

- Experiment with advanced time-series forecasting architectures
- Incorporate weather and operational variables
- Implement automated model evaluation and tracking
- Build multi-step forecasting
- Improve deployment and monitoring workflows

---

**Author:** Harsh Dixit

[GitHub Repository](https://github.com/harshdixit05/energy-forecasting)