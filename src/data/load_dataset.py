import pandas as pd
import os

from config import core


def load():
    """
    Function that imports the raw data and returns a pandas dataframe containing the datset
    :return: df, dtype: object
    """

    # define the dataset path
    filepath = os.path.join(core.RAW_DIR, core.DATASET_NAME)

    # load the dataset and save as a python dataframe
    df = pd.read_csv(filepath)

    return df