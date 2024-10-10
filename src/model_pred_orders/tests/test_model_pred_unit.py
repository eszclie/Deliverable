import numpy as np

from .. import predict_orders


def test_make_predictions():
    # Fixed model path
    model_path = "src/model/XGBRegressor"
    model = predict_orders.load_model(model_path)

    # Sample input data for prediction in the expected format
    inputs = [["2024-09-27", 45], ["2024-09-28", 55]]

    # Make predictions
    orders_pred = predict_orders.make_predictions(model, inputs)

    # Assertions to check the type and shape of orders_pred
    assert isinstance(orders_pred, np.ndarray), "Predictions should be a numpy array"
    assert len(orders_pred) == len(inputs), "Number of predictions should match number of inputs"

    # Check that all predicted orders are non-negative floats
    for predicted in orders_pred:
        assert isinstance(predicted, (float, np.float32)), f"Predicted order {predicted} is not a float"
        assert predicted >= 0, f"Predicted order {predicted} is negative"
