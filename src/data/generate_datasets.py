"""
generate_datasets.py

This module generates datasets, features, labels, training and test splits for modelling.
"""
import numpy as np
from sklearn.model_selection import train_test_split

from src.data.load_dataset import load
from src.features.feature_exploration import xy_separator

# Load the raw dataset
data = load()

# Separate the dataset into features and labels
data_X, data_y = data[xy_separator()[0]].values, data[xy_separator()[1]].values

# Split dataset into training and test sets, 70-30%
X_train, X_test, y_train, y_test = train_test_split(data_X,
                                                    data_y,
                                                    test_size=0.3,
                                                    random_state=0,
                                                    stratify=data_y)