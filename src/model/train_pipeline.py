import joblib
import os

from config import core
from src.data.generate_datasets import X_train, y_train
from src.model.pipeline import wine_pipeline


def train():
    """Train the multiclass classifier model"""

    # fit the pipeline to train a Support Vector Machine on train set
    multi_model = wine_pipeline.fit(X_train, y_train)

    return multi_model


def save_model(trained_model) -> None:
    """Save the trained pipeline in the model folder"""

    # Define the save path
    save_name = 'wine_multi_model.pkl'
    save_path = os.path.join(core.MODEL_DIR, save_name)

    # Save the trained pipeline/multiclass model
    joblib.dump(trained_model, save_path)


wine_multi_model = train()
save_model(wine_multi_model)
