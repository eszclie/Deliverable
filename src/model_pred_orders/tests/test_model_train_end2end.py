from xgboost import XGBRegressor

from .. import train_model


def test_train_model():
    # Act
    x, y = train_model.load_data()
    model = train_model.train_model(x, y)

    # Assert
    assert not x.empty
    assert not y.empty
    assert isinstance(model, XGBRegressor)
    assert hasattr(model, "feature_importances_")

    # Additional assertions for model feature importance
    assert model.feature_importances_ is not None
    assert model.feature_importances_.shape[0] == x.shape[1]
