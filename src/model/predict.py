"""
predict.py

Makes a prediction using the trained multiclass classifier
"""

import joblib
import os

from config import core
from src.data.generate_datasets import X_test

# load the saved model pipeline
filename = os.path.join(core.MODEL_DIR, 'wine_multi_model.pkl')
multi_model = joblib.load(filename)

# make prediction
wine_predictions = multi_model.predict(X_test)
