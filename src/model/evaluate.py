"""
evaluate.py

Evaluates the trained multiclass classifier with scikit-learn's metrics
"""
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

from config import core
from src.model.predict import multi_model, wine_predictions
from src.data.generate_datasets import X_test, y_test


def overall_metrics(test, pred, prob) -> None:
    """
    Evaluates the following metrics to ensure that the model makes accurate predictions;
        Accuracy score,
        Precision score,
        Recall score

    This function prints the metrics above and saves them to a text file at the reports folder

    :param test: the test set of the labels
    :param pred: the predictions made using the trained multiclass classifier
    :param prob: the variety class probability scores
    """

    # Save the metrics to the reports location
    save_loc = os.path.join(core.OVR_DIR, 'overall_metrics.txt')
    with open(save_loc, 'w') as f:
        print("Overall Accuracy:", accuracy_score(test, pred), file=f)
        print("Overall Precision:", precision_score(test, pred, average='macro'), file=f)
        print("Overall Recall:", recall_score(test, pred, average='macro'), file=f)
        print('Average AUC:', roc_auc_score(test, prob, multi_class='ovr'), file=f)


def confusion_matrix_plot(mcm, classes) -> None:
    """
    Visualizes the confusion matrix of the trained model in a heat map

    :param mcm :np.ndarray, a 3d confusion matrix
    :param classes: the variety classes of the wine dataset(1, 2, 3)
    """

    # plot the confusion matrix in a heat map
    plt.imshow(mcm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.xlabel("Predicted Cultivars")
    plt.ylabel("Actual Cultivars")

    # Save the metrics to the reports location
    save_loc = os.path.join(core.CM_DIR, 'confusion_matrix.png')
    plt.savefig(save_loc)

    plt.show()


def main():
    """Run the evaluation reports"""

    # get predictions from text data
    predictions = wine_predictions
    probabilities = multi_model.predict_proba(X_test)

    # evaluate the confusion matrix
    mcm = confusion_matrix(y_test, wine_predictions)

    # define class identifiers
    classes = ['Variety 0', 'Variety 1', 'Variety 2']

    # output the overall metrics
    overall_metrics(test=y_test, pred=predictions, prob=probabilities)

    # plot the confusion matrix
    confusion_matrix_plot(mcm=mcm, classes=classes)


main()
