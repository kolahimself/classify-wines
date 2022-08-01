"""
feature_exploration.py

This module separates features & labels from the dataset,
saves images of box plots showing data profiles for every feature in the dataset
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

from src.data.load_dataset import load
from config import core


def xy_separator():
    """
    Separates features and labels from the dataset

    :return x, y: Features and label extracted from the dataframe
    """

    # load the dataset
    data = load()

    # Separate features and labels
    x = list(data.columns[1:])
    y = 'Cultivar'

    return x, y


def save_plots() -> None:
    """
    Creates box plots from the features of the dataset
    """

    # import the loaded dataset
    data = load()

    # define the features and labels
    data_features, data_labels = xy_separator()

    for cols in data_features:
        # create the boxplot for each feature
        data.boxplot(column=cols, by=data_labels, figsize=(6, 6))
        plt.title(cols)

        # save the plots to the images/explorations folder
        save_name = str(cols) + '.png'
        save_path = os.path.join(core.EXPLORATIONS_DIR, save_name)
        plt.savefig(save_path)


save_plots()