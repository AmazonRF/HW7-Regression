"""
Write your unit tests here. Some tests to include are listed below.
This is not an exhaustive list.

- check that prediction is working correctly
- check that your loss function is being calculated correctly
- check that your gradient is being calculated correctly
- check that your weights update during training
"""

# Imports
import pytest
from regression import loadDataset, LogisticRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np

def test_prediction():
    # Loading the dataset and splitting it into training and validation sets
	X_train, X_val, y_train, y_val = loadDataset(split_percent=0.8)

	sc = StandardScaler()
    # Standardizing the training data
	X_train = sc.fit_transform(X_train)
    # Applying the same transformation to the validation data
	X_val = sc.transform(X_val)

	log_model = LogisticRegressor(num_feats=6)
	X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
    # Making predictions using the logistic regression model
	predict_res = log_model.make_prediction(X_val)
    # Asserting that every input has a corresponding prediction
	assert len(predict_res) == len(X_val), "Each test data should have one test result"
    # Ensuring all predictions are probabilities between 0 and 1
	assert np.max(predict_res) <= 1, "The predict result should be [0,1]"
	assert np.min(predict_res) >= 0, "The predict result should be [0,1]"

def test_loss_function():
    # Repeating the dataset loading and splitting process
	X_train, X_val, y_train, y_val = loadDataset(split_percent=0.8)

	sc = StandardScaler()
	X_train = sc.fit_transform(X_train)
	X_val = sc.transform(X_val)

	log_model = LogisticRegressor(num_feats=6)
	X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
	predict_res = log_model.make_prediction(X_val)
    # Calculating the loss using the true labels and the predicted probabilities
	predict_true_loss = log_model.loss_function(y_val, predict_res)
    # Ensuring the loss is non-negative
	assert predict_true_loss >= 0

    # Testing if passing the true labels as predictions causes a RuntimeWarning
	with pytest.warns(RuntimeWarning) as record:
		assert np.isnan(log_model.loss_function(y_val, y_val))

def test_gradient():
    # Loading dataset and preparing it
	X_train, X_val, y_train, y_val = loadDataset(split_percent=0.8)

	sc = StandardScaler()
	X_train = sc.fit_transform(X_train)
	X_val = sc.transform(X_val)

	log_model = LogisticRegressor(num_feats=6)
	X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
    # Calculating the gradient of the loss function with respect to the weights
	grad = log_model.calculate_gradient(y_val, X_val)
    # Checking if the gradient vector's length matches the weight vector's length
	assert len(grad) == log_model.W.shape[0], "Output gradient shape should same with weight matrix"
    # Ensuring the gradient vector consists of floating-point numbers
	assert grad.dtype == np.float64, "Output gradient data type should be float64"

def test_training():
    # Loading and preparing the dataset
	X_train, X_val, y_train, y_val = loadDataset(split_percent=0.8)
	
	sc = StandardScaler()
	X_train = sc.fit_transform(X_train)
	X_val = sc.transform(X_val)
	
	log_model = LogisticRegressor(num_feats=6)
    # Recording the initial weights
	start_W = log_model.W
    # Training the model using the training data
	log_model.train_model(X_train, y_train, X_val, y_val)
    # Recording the weights after training
	out_W = log_model.W
    # Asserting that the weights have changed as a result of training
	assert np.sum(start_W) != np.sum(out_W), "Weights should be changed after training"
