import pytest
# TODO: add necessary import
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from ml.model import train_model, inference, compute_model_metrics
# TODO: implement the first test. Change the function name and input as needed
def test_one():
    """
    Test that train_model returns a trained RandomForestClassifier.
    """
    X = np.random.rand(20, 5)
    y = np.random.randint(0, 2, size=20)

    model = train_model(X, y)

    assert model is not None
    assert isinstance(model, RandomForestClassifier)



# TODO: implement the second test. Change the function name and input as needed
def test_two():
    """
    Test that inference returns predictions with the same length as input
    """
    X = np.random.rand(15, 5)
    y = np.random.randint(0, 2, size=15)

    model = train_model(X, y)
    preds = inference(model, X)

    assert preds is not None
    assert len(preds) == len(X)
    


# TODO: implement the third test. Change the function name and input as needed
def test_three():
    """
    Test that compute_model_metrics returns float values.
    """
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])

    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)

    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)
