"""
This module provides classes for managing and validating configuration parameters
for various machine learning models, specifically focusing on ARIMA-based models.
The module includes a base class that standardizes the process of handling model
configuration, ensuring that all derived classes follow a consistent pattern for
validating and setting parameters.

Classes
-------
AbstractBaseModelConfig : ABC
    An abstract base class for the `BaseModelConfig` class.

BaseModelConfig : AbstractBaseModelConfig
    A base class that provides common functionality for model
    configuration, including parameter validation, default value handling, and
    error checking. Subclasses are required to define a `SUPPORTED_PARAMETERS`
    dictionary that specifies the expected parameter types, default values,
    and any valid choices.

ARIMAConfig : BaseModelConfig
    A configuration class for ARIMA model parameters. Inherits from `BaseModelConfig`
    and defines specific parameters used by ARIMA and ARIMA_PLUS models. This class
    ensures that the parameters adhere to the expected types and valid choices.

ARIMA_PLUS_XREG_Config : BaseModelConfig
    A configuration class for ARIMA_PLUS_XREG model parameters. This class extends
    `BaseModelConfig` and includes additional parameters for handling exogenous
    variables (`xreg_features`) and other settings specific to the `ARIMA_PLUS_XREG` model.

Usage
-----
These configuration classes are intended to be used in the setup and validation of
model parameters before they are passed to machine learning model training functions.
By leveraging these classes, developers can ensure that all configuration parameters
are correctly typed, fall within valid ranges, and adhere to expected choices, reducing
the likelihood of runtime errors.

Example
-------
>>> config = ARIMAConfig(model_type="ARIMA")
>>> print(config.model_type)
'ARIMA'

>>> xreg_config = ARIMA_PLUS_XREG_Config(
...     model_type="ARIMA_PLUS_XREG",
...     xreg_features=["feature1", "feature2"],
...     non_negative_forecast=True
... )
>>> print(xreg_config.xreg_features)
['feature1', 'feature2']
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, List


class AbstractBaseModelConfig(ABC):  # pylint: disable=too-few-public-methods
    """Abstract base class for `BaseModelConfig` configuration class."""

    @property
    @abstractmethod
    def SUPPORTED_PARAMETERS(self) -> Dict[  # pylint: disable=invalid-name
        str, Tuple[Any, Any, List[Any]]
    ]:
        """
        This abstract property must be implemented by subclasses.
        It should return a dictionary where the keys are parameter names,
        and the values are tuples containing the expected type, default value,
        and a list of valid choices (if any).
        """


class BaseModelConfig(AbstractBaseModelConfig):
    """
    Base class for model configuration parameters.

    This class provides common functionality for handling configuration parameters
    passed via kwargs, including unpacking, validation, and setting default values.

    Subclasses must define the `SUPPORTED_PARAMETERS` dictionary, which specifies
    the expected parameter types, default values, and any restricted choices.
    """

    @property
    def SUPPORTED_PARAMETERS(self) -> Dict[str, Tuple[Any, Any, List[Any]]]:
        return {}

    def __init__(self, **kwargs):
        self._params = {}
        self._validate_and_set_parameters(kwargs)

    def _validate_and_set_parameters(self, kwargs: Dict[str, Any]):
        for key, (expected_type, default_value, choices) in self.SUPPORTED_PARAMETERS.items():
            if key in kwargs:
                value = kwargs[key]
                if not isinstance(value, expected_type):
                    raise ValueError(
                        f"Invalid value for parameter '{key}': expected {expected_type.__name__}, "
                        f"but got {type(value).__name__}."
                    )
                if choices and value not in choices:
                    raise ValueError(
                        f"Invalid value for parameter '{key}': got '{value}', "
                        f"but expected one of {choices}."
                    )
                self._params[key] = value
            else:
                self._params[key] = default_value

        # Identify unsupported parameters
        unsupported_params = set(kwargs) - set(self.SUPPORTED_PARAMETERS)
        if unsupported_params:
            raise ValueError(
                f"Unsupported parameters provided: {', '.join(unsupported_params)}. "
                "Please check your input."
            )

    def __getattr__(self, name: str) -> Any:
        if name in self._params:
            return self._params[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")


class ARIMAConfig(BaseModelConfig):  # pylint: disable=too-few-public-methods
    """
    Configuration class for `'ARIMA'` model parameters.

    Inherits common functionality from `BaseModelConfig` and defines specific
    parameters for `'ARIMA'` models, including validation of choices for some
    parameters.
    """

    @property
    def SUPPORTED_PARAMETERS(self) -> Dict[str, Tuple[Any, Any, List[Any]]]:
        return {
            "model_type": (str, "ARIMA_PLUS", ["ARIMA_PLUS", "ARIMA"]),
            "auto_arima": (bool, True, []),
            "forecast_limit_lower_bound": (int, 0, []),
            "clean_spikes_and_dips": (bool, True, []),
            "decompose_time_series": (bool, True, []),
            "holiday_region": (str, "US", []),
            "data_frequency": (str, "AUTO_FREQUENCY",
                               ["AUTO_FREQUENCY", "DAILY", "WEEKLY", "MONTHLY"]),
            "adjust_step_changes": (bool, True, []),
        }


class ARIMA_PLUS_XREG_Config(BaseModelConfig):  # pylint: disable=invalid-name, too-few-public-methods
    """
    Configuration class for `'ARIMA_PLUS_XREG'` model parameters.

    Inherits common functionality from `BaseModelConfig` and defines specific
    parameters for `'ARIMA_PLUS_XREG'` models, including validation of choices for
    some parameters.
    """

    @property
    def SUPPORTED_PARAMETERS(self) -> Dict[str, Tuple[Any, Any, List[Any]]]:
        return {
            "model_type": (str, "ARIMA_PLUS_XREG", ["ARIMA_PLUS_XREG"]),
            "auto_arima": (bool, True, []),
            "clean_spikes_and_dips": (bool, True, []),
            "holiday_region": (str, "US", []),
            "data_frequency": (str, "AUTO_FREQUENCY",
                               ["AUTO_FREQUENCY", "DAILY", "WEEKLY", "MONTHLY"]),
            "adjust_step_changes": (bool, True, []),
            "non_negative_forecast": (bool, False, []),
        }
