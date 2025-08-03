# MNIST_dataset_project
It is a machine learning project for MNIST dataset. This project classifies images into digits from 0-9
MNIST Handwritten Digit Classification
This repository contains a Jupyter notebook demonstrating a multilabel classification project on the MNIST dataset using a K-Nearest Neighbors (KNN) classifier.

Project Goal
The objective of this project is to build and evaluate a model capable of classifying handwritten digits (0-9) from the MNIST dataset.

Dataset
The project uses the widely-known MNIST dataset, consisting of 70,000 grayscale images of handwritten digits (28x28 pixels). The dataset is split into a training set of 60,000 images and a test set of 10,000 images.

Approach
The project follows these key steps:

Data Loading and Exploration: Loading the MNIST dataset using sklearn.datasets.fetch_openml and examining its structure.
Data Preparation: Splitting the data into training and testing sets and creating multilabel target variables where each sample is represented by a boolean array indicating the presence of each digit class (0-9).
Model Training: Training a K-Nearest Neighbors (KNeighborsClassifier) model on the training data.
Model Evaluation: Evaluating the model's performance on the test set using various metrics suitable for multilabel classification, including:
Macro F1 Score
Macro Precision
Macro Recall
Accuracy
Per-class Confusion Matrices
Results
The trained KNN model achieved strong performance on the MNIST test set. Key evaluation metrics include:

Macro F1 Score: [Insert your Macro F1 Score here, e.g., ~0.97]
Macro Precision: [Insert your Macro Precision here]
Macro Recall: [Insert your Macro Recall here]
Accuracy: [Insert your Accuracy here]
Individual confusion matrices for each digit class provide a detailed view of the model's performance.

Files
[Your_Notebook_Name].ipynb: The Jupyter notebook containing the code for this project.
knn_mnist_multilabel_model.joblib (Optional, if uploaded): The exported trained model.
Requirements
The code requires the following Python libraries:

numpy
pandas
matplotlib
sklearn
joblib (for model export/import)
