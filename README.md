# Predictive Maintenance Dashboard

An end-to-end machine learning project for predicting industrial machine failures using the **AI4I 2020 Predictive Maintenance Dataset**. The project uses **XGBoost** with feature engineering and class-imbalance handling, along with an interactive **Streamlit dashboard** for machine health and failure-risk prediction.

## Live Demo

[Open the Predictive Maintenance Dashboard](https://thanvishpredmain.streamlit.app/)

## Overview

Unexpected machine failures can lead to production downtime, maintenance costs, and equipment damage. Predictive maintenance uses machine operating data to identify patterns associated with potential failures before they occur.

This project implements the following pipeline:

**Data → Feature Engineering → XGBoost Model → Evaluation → Streamlit Dashboard**

The dashboard allows users to enter machine operating parameters and receive a predicted failure probability and equipment health status.

## Features

* XGBoost-based machine failure prediction
* AI4I 2020 Predictive Maintenance Dataset
* Feature engineering using:

  * Temperature Difference
  * Power
* Class imbalance handling using `scale_pos_weight`
* Chronological 80/20 train-test split
* Model evaluation using:

  * Precision
  * Recall
  * F1 Score
  * ROC-AUC
* Interactive Streamlit dashboard
* Saved trained XGBoost model for application inference
* Separate model metadata for feature and evaluation information

## Dataset

The project uses the **AI4I 2020 Predictive Maintenance Dataset** from the UCI Machine Learning Repository.

The dataset contains machine operating information including:

* Air Temperature
* Process Temperature
* Rotational Speed
* Torque
* Tool Wear
* Machine Type
* Machine Failure

### Model Features

| Feature            | Description                                    |
| ------------------ | ---------------------------------------------- |
| `Air_Temp`         | Air temperature in Kelvin                      |
| `Process_Temp`     | Process temperature in Kelvin                  |
| `Rotational_Speed` | Machine rotational speed in rpm                |
| `Torque`           | Machine torque in Nm                           |
| `Tool_Wear`        | Tool wear duration in minutes                  |
| `Type_L`           | Encoded low-quality machine type               |
| `Type_M`           | Encoded medium-quality machine type            |
| `Temp_Diff`        | Difference between process and air temperature |
| `Power`            | Torque multiplied by rotational speed          |

## Feature Engineering

Two additional features are created to provide the model with useful machine-operating relationships.

### Temperature Difference

```text
Temp_Diff = Process_Temp - Air_Temp
```

This represents the difference between the machine's process temperature and surrounding air temperature.

### Power

```text
Power = Torque × Rotational_Speed
```

This provides an additional representation of the mechanical load on the machine.

## Machine Learning Model

The project uses an **XGBoost Classifier** for binary machine-failure prediction.

Machine failures are less frequent than normal operating conditions, creating a class imbalance problem. To address this, the model uses:

```python
scale_pos_weight
```

This gives greater importance to the minority failure class and helps the model detect potential failures more effectively.

The dataset is divided chronologically into:

* **80%** training data
* **20%** testing data

A chronological split is used to provide a more realistic evaluation scenario by training on earlier observations and testing on later observations.

## Model Performance

The trained model is evaluated using metrics that are useful for failure detection.

| Metric    |     Result |
| --------- | ---------: |
| Precision | **48.15%** |
| Recall    | **66.67%** |
| F1 Score  | **55.91%** |
| ROC-AUC   | **96.64%** |

### Evaluation Metrics

**Precision** measures how many machines predicted as failures are actually failures.

**Recall** measures how many of the actual machine failures are successfully detected.

**F1 Score** provides a balance between precision and recall.

**ROC-AUC** measures how well the model distinguishes between normal and failure conditions across different classification thresholds.

For predictive maintenance, recall is particularly important because failing to detect an actual machine failure can be more costly than generating an additional inspection alert.

## Streamlit Dashboard

The Streamlit application provides an interactive interface where users can enter machine operating parameters such as:

* Air Temperature
* Process Temperature
* Rotational Speed
* Torque
* Tool Wear
* Machine Type

The application calculates the engineered features and passes the resulting feature vector to the trained XGBoost model.

The dashboard then displays:

* Predicted failure probability
* Equipment health status
* Calculated temperature difference
* Calculated power metric

### Health Status

The application categorizes the predicted risk into three levels:

* **NORMAL** — low predicted failure risk
* **WARNING** — elevated predicted failure risk
* **CRITICAL** — high predicted failure risk requiring attention

These categories are intended as decision-support indicators and do not replace professional maintenance procedures.

## Project Structure

```text
predictive_maintenance_app/
│
├── predictive_maintenance.ipynb
├── predictive_maintenance_app.py
├── predictive_maintenance_model.json
├── model_metadata.json
├── requirements.txt
└── README.md
```

### File Description

| File                                | Purpose                                                                |
| ----------------------------------- | ---------------------------------------------------------------------- |
| `predictive_maintenance.ipynb`      | Data preprocessing, feature engineering, model training and evaluation |
| `predictive_maintenance_app.py`     | Streamlit dashboard                                                    |
| `predictive_maintenance_model.json` | Trained XGBoost model                                                  |
| `model_metadata.json`               | Model features and evaluation metadata                                 |
| `requirements.txt`                  | Python dependencies                                                    |
| `README.md`                         | Project documentation                                                  |

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Thanvish7/predictive_maintenance_app.git
cd predictive_maintenance_app
```

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Run the Streamlit application

```bash
python -m streamlit run predictive_maintenance_app.py
```

The application will open in your browser.

## Requirements

The main technologies used in this project are:

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Streamlit
* Jupyter Notebook
* Matplotlib
* Seaborn

All required packages can be installed using:

```bash
python -m pip install -r requirements.txt
```

## Model Workflow

```text
AI4I 2020 Dataset
        ↓
Data Cleaning
        ↓
Categorical Encoding
        ↓
Feature Engineering
        ↓
Chronological Train/Test Split
        ↓
XGBoost Classifier
        ↓
Model Evaluation
        ↓
Saved Model
        ↓
Streamlit Dashboard
        ↓
Machine Failure Prediction
```

## Limitations

* The model is trained on the AI4I 2020 dataset and may not generalize directly to every industrial machine.
* The dashboard currently accepts machine telemetry through user inputs rather than a live physical sensor connection.
* Predictions should be treated as decision-support information rather than a replacement for professional maintenance inspection.
* Model performance depends on the quality and distribution of the input data.

## Future Improvements

* Connect the dashboard to real-time IoT sensor data.
* Add historical machine-health trends.
* Implement automated maintenance alerts.
* Add model explainability using SHAP.
* Store prediction history for further analysis.
* Experiment with additional machine-learning models.
* Integrate the system with a real-time industrial IoT pipeline.

## Author

**Thanvish A**

B.Tech – Computer Science (IoT and Automation)
SASTRA University

## Project Repository

[GitHub Repository](https://github.com/Thanvish7/predictive_maintenance_app)
