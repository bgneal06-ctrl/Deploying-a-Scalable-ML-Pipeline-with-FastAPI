import pytest
# TODO: add necessary import
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ml.data import process_data
from ml.model import train_model, inference, save_model, load_model 


# Setup reusable test data 
@pytest.fixture
def sample_data():
    """Create sample DataFrame similar to census data."""
    data= pd.DataFrame({
        "age": [25, 40, 35, 28],
        "workclass": ["Private", "Self-emp", "Private", "Government"],
        "education": ["Bachelors", "Masters", "HS-grad", "Bachelors"],
        "marital-status": ["Single", "Married", "Married", "Single"],
        "occupation": ["Tech-support", "Exec-managerial", "Sales", "Clerical"],
        "relationship": ["Not-in-family", "Husband", "Wife", "Unmarried"],
        "race": ["White", "White", "Black", "Asian-Pac-Islander"],
        "sex": ["Male", "Female", "Female", "Male"],
        "hours-per-week": [40, 50, 45, 60],
        "native-country": ["United-States"] * 4,
        "salary": ["<=50K", ">50K", "<=50K", ">50K"]
    })
    categorical_features = [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country"
    ]
    return data, categorical_features



# Test 1: Data Processing
def test_process_data_output_shapes(sample_data):
    """
    Test that process_data returns arrays of the correct shape.
    """
    data, cat_features = sample_data
    X, y, encoder, lb = process_data(
        data,
        categorical_features=cat_features,
        label="salary",
        training=True
    )
    assert X.shape[0] == y.shape[0], "Feature and label row counts should match"
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)


# Test 2: Model Training and Inference
def test_train_and_inference(sample_data):
    """
    Test the a model can be trained and produces predictions.
    """
    data, cat_features = sample_data
    X, y, encoder, lb = process_data(
        data,
        categorical_features=cat_features,
        label="salary",
        training=True
    )
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier), "Model should be a RandomForestClassifier"
    preds = inference(model, X)
    assert len(preds) == len(y), "Predictions should match number of samples"
    assert set(np.unique(preds)).issubset({0, 1}), "Predictions must be binary"

# Test 3: Save and Load Model
"""
Test that a saved model can be loaded back and still make valid predictions.
"""
def test_save_and_load_model(tmp_path, sample_data):
    data, cat_features = sample_data
    X, y, encoder, lb = process_data(
        data,
        categorical_features=cat_features,
        label="salary",
        training=True
    )
    model = train_model(X, y)
    model_path = tmp_path / "model.pkl"

    save_model(model, model_path)
    loaded_model = load_model(model_path)

    preds_original = inference(model, X)
    preds_loaded = inference(loaded_model, X)

    assert np.array_equal(preds_original, preds_loaded), "Loaded model predictions must match original"
