""" Configuration file.
All static variables can be assigned in these settings.py file
"""

import os

# Directories
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(os.getcwd())))

# > data
DATA_DIR = os.path.join(ROOT_DIR, 'data')
EXTERNAL_DIR = os.path.join(DATA_DIR, 'external')
INTERIM_DIR = os.path.join(DATA_DIR, 'interim')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
RAW_DIR = os.path.join(DATA_DIR, 'raw')

# > images
IMAGES_DIR = os.path.join(ROOT_DIR, 'images')
EXPLORATIONS_DIR = os.path.join(IMAGES_DIR, 'explorations')

# > models
MODEL_DIR = os.path.join(ROOT_DIR, 'models')

# > reports
REPORTS_DIR = os.path.join(ROOT_DIR, 'reports')
OVR_DIR = os.path.join(REPORTS_DIR, 'overall_metrics')
CM_DIR = os.path.join(REPORTS_DIR, 'confusion_matrix')

# Dataset file name
DATASET_NAME = 'wine.csv'
