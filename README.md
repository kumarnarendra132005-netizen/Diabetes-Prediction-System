# Diabetes Prediction System

A machine learning-based diabetes prediction system developed using Python and TensorFlow. The project uses patient health-related features to predict whether an individual belongs to the diabetes-positive or diabetes-negative class.

## Project Overview

This project implements a binary classification neural network using the Pima Indians Diabetes Dataset.

The system takes the following patient attributes as input:

- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age

The target variable is `Outcome`:

- `0` - No Diabetes
- `1` - Diabetes

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- TensorFlow
- Keras

## Machine Learning Workflow

1. Dataset loading
2. Data inspection
3. Missing-value checking
4. Feature and target separation
5. Train-test splitting
6. Feature standardization using StandardScaler
7. Neural network construction
8. Model training
9. Model evaluation
10. Confusion matrix and performance analysis
11. Model saving

## Neural Network Architecture

```text
Input Layer: 8 Features
        ↓
Dense Layer: 32 Neurons + ReLU
        ↓
Dense Layer: 16 Neurons + ReLU
        ↓
Output Layer: 1 Neuron + Sigmoid