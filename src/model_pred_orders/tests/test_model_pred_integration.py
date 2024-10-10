from xgboost import XGBRegressor

from .. import predict_orders


def test_load_model():
    """Test if the model is loaded correctly."""
    model_path = "src/model/XGBRegressor"
    model = predict_orders.load_model(model_path)

    # Assert that the model is not None
    assert model is not None, "Model failed to load"

    # Assert that the model is a scikit-learn LinearRegression instance
    assert isinstance(model, XGBRegressor)
