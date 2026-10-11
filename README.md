# Three-Phase Transmission Line Fault Detection

## Project Overview

This project compares logistic regression implemented using NumPy with Support Vector Machines (SVMs) for detecting faults in a three-phase transmission line dataset.

## Dataset

The dataset contains three-phase current measurements, voltage measurements, and a binary fault label.

## Methods

- Exploratory Data Analysis (EDA)
- Logistic Regression using batch gradient descent
- Feature engineering with squared features (Set Q)
- Linear and RBF Support Vector Machines
- Test-set evaluation using accuracy, confusion matrices, and missed faults

## Project Structure

- `data/` - Original dataset and prepared splits
- `src/` - Data preparation, training, and evaluation scripts
- `results/` - Plots, metrics, and saved models
- `app/` - Streamlit prediction interface

## Installation

Install the required Python packages:

```bash
pip install numpy pandas matplotlib scikit-learn joblib streamlit
```

## Running the Project

Run these commands from the project root, in the following order:

```bash
python src/prepare.py
python src/eda.py
python src/train_logistic.py
python src/train_svm.py
python src/test_models.py
streamlit run app/app.py
```

## Model Evaluation

The models are evaluated using a held-out test set. Evaluation includes accuracy, confusion matrices, and missed fault detections.

## Limitations

This project is for educational purposes only. Its predictions are not a substitute for validated electrical protection systems.
