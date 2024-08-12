"""
Unit tests for the classes defined in the `iowa_forecast.models_configs` module.

This test suite verifies that the `ARIMAConfig` class behaves as expected, including:
- Correct handling of default values when no parameters are provided.
- Proper assignment of custom values passed through kwargs.
- Validation of parameter types, ensuring that values match the expected types.
- Enforcement of valid choices for parameters with restricted options.
- Error handling for unsupported parameters and invalid attribute access.

These tests are designed to ensure robustness and correctness in the configuration of
ARIMA model parameters.
"""
import pytest
from iowa_forecast.models_configs import ARIMAConfig, ARIMA_PLUS_XREG_Config, BaseModelConfig


def test_arima_config_defaults():
    """Test that ARIMAConfig sets the default values correctly."""
    config = ARIMAConfig()

    assert config.model_type == "ARIMA_PLUS"
    assert config.auto_arima is True
    assert config.forecast_limit_lower_bound == 0
    assert config.clean_spikes_and_dips is True
    assert config.decompose_time_series is True
    assert config.holiday_region == "US"
    assert config.data_frequency == "AUTO_FREQUENCY"
    assert config.adjust_step_changes is True


def test_arima_config_custom_values():
    """Test that ARIMAConfig correctly assigns custom values."""
    config = ARIMAConfig(
        model_type="ARIMA",
        auto_arima=False,
        forecast_limit_lower_bound=10,
        clean_spikes_and_dips=False,
        decompose_time_series=False,
        holiday_region="UK",
        data_frequency="DAILY",
        adjust_step_changes=False
    )

    assert config.model_type == "ARIMA"
    assert config.auto_arima is False
    assert config.forecast_limit_lower_bound == 10
    assert config.clean_spikes_and_dips is False
    assert config.decompose_time_series is False
    assert config.holiday_region == "UK"
    assert config.data_frequency == "DAILY"
    assert config.adjust_step_changes is False


def test_arima_config_invalid_type():
    """Test that ARIMAConfig raises a ValueError for invalid types."""
    with pytest.raises(ValueError) as exc_info:
        ARIMAConfig(model_type=123)  # model_type should be a string
    assert "Invalid value for parameter 'model_type'" in str(exc_info.value)

    with pytest.raises(ValueError) as exc_info:
        ARIMAConfig(auto_arima="yes")  # auto_arima should be a boolean
    assert "Invalid value for parameter 'auto_arima'" in str(exc_info.value)

    with pytest.raises(ValueError) as exc_info:
        ARIMAConfig(forecast_limit_lower_bound="low")  # should be an int
    assert "Invalid value for parameter 'forecast_limit_lower_bound'" in str(
        exc_info.value)


def test_arima_config_invalid_choice():
    """Test that ARIMAConfig raises a ValueError for invalid choices."""
    with pytest.raises(ValueError) as exc_info:
        ARIMAConfig(model_type="INVALID_TYPE")  # Invalid choice for model_type
    assert "Invalid value for parameter 'model_type'" in str(exc_info.value)
    assert "but expected one of ['ARIMA_PLUS', 'ARIMA']" in str(exc_info.value)


def test_arima_config_unsupported_parameter():
    """Test that ARIMAConfig raises a ValueError for unsupported parameters."""
    with pytest.raises(ValueError) as exc_info:
        ARIMAConfig(unsupported_param="value")
    assert "Unsupported parameters provided: unsupported_param" in str(exc_info.value)


def test_arima_plus_xreg_config_defaults():
    """Test that ARIMA_PLUS_XREG_Config sets the default values correctly."""
    config = ARIMA_PLUS_XREG_Config()

    assert config.model_type == "ARIMA_PLUS_XREG"
    assert config.auto_arima is True
    assert config.clean_spikes_and_dips is True
    assert config.holiday_region == "US"
    assert config.data_frequency == "AUTO_FREQUENCY"
    assert config.adjust_step_changes is True
    assert config.non_negative_forecast is False


def test_arima_plus_xreg_config_custom_values():
    """Test that ARIMA_PLUS_XREG_Config correctly assigns custom values."""
    config = ARIMA_PLUS_XREG_Config(
        auto_arima=False,
        clean_spikes_and_dips=False,
        holiday_region="EU",
        data_frequency="WEEKLY",
        adjust_step_changes=False,
        non_negative_forecast=True
    )

    assert config.model_type == "ARIMA_PLUS_XREG"
    assert config.auto_arima is False
    assert config.clean_spikes_and_dips is False
    assert config.holiday_region == "EU"
    assert config.data_frequency == "WEEKLY"
    assert config.adjust_step_changes is False
    assert config.non_negative_forecast is True


def test_arima_plus_xreg_config_invalid_type():
    """Test that ARIMA_PLUS_XREG_Config raises a ValueError for invalid types."""
    with pytest.raises(ValueError) as exc_info:
        ARIMA_PLUS_XREG_Config(non_negative_forecast="yes")  # should be a boolean
    assert "Invalid value for parameter 'non_negative_forecast'" in str(exc_info.value)


def test_arima_plus_xreg_config_invalid_choice():
    """Test that ARIMA_PLUS_XREG_Config raises a ValueError for invalid choices."""
    with pytest.raises(ValueError) as exc_info:
        ARIMA_PLUS_XREG_Config(
            model_type="INVALID_TYPE")  # Invalid choice for model_type
    assert "Invalid value for parameter 'model_type'" in str(exc_info.value)
    assert "but expected one of ['ARIMA_PLUS_XREG']" in str(exc_info.value)


def test_arima_plus_xreg_config_unsupported_parameter():
    """Test that ARIMA_PLUS_XREG_Config raises a ValueError for unsupported parameters."""
    with pytest.raises(ValueError) as exc_info:
        ARIMA_PLUS_XREG_Config(unsupported_param="value")
    assert "Unsupported parameters provided: unsupported_param" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main()
