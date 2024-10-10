# %%
import numpy as np

import model_pred_orders.predict_orders as predict_orders

# %%


def test_make_predictions():
    """Test the make_predictions function end-to-end."""
    # Load the model
    model_path = "src/model/XGBRegressor"
    model = predict_orders.load_model(model_path)

    # Sample input data for testing
    inputs = [["2024-09-27", 45], ["2024-09-28", 55]]

    # Make predictions
    predictions = predict_orders.make_predictions(model, inputs)

    # Assert that the predictions are of type numpy array
    assert isinstance(predictions, np.ndarray), "Predictions should be a numpy array"

    # Assert the number of predictions matches the number of inputs
    assert len(predictions) == len(inputs), "Number of predictions should match number of inputs"

    # Assert that predictions are non-negative and floats
    for prediction in predictions:
        assert isinstance(prediction, (float, np.float32, np.float64)), "Prediction should be a float"
        assert prediction >= 0, f"Prediction {prediction} should be non-negative"
