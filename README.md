# Diabetes Prediction System

This project implements a binary classification neural network using the Pima Indians Diabetes Dataset.

The system uses the following patient attributes as input:

- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age

The target variable is `Outcome`:

- `0` - No Diabetes
- `1` - Diabetes

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow
- Keras

## Machine Learning Workflow

1. Dataset loading
2. Data inspection
3. Missing-value checking
4. Statistical data analysis
5. Feature and target separation
6. Train-test splitting
7. Feature standardization using StandardScaler
8. Neural network construction
9. Model compilation
10. Model training
11. Model evaluation

## Neural Network Architecture

```text
Input Layer: 8 Features
        ↓
Dense Layer: 32 Neurons (ReLU)
        ↓
Dense Layer: 16 Neurons (ReLU)
        ↓
Output Layer: 1 Neuron (Sigmoid)