# stdlib
from typing import Any

# synthcity relative
from .tte_aft import WeibullAFTTimeToEvent
from .tte_coxph import CoxPHTimeToEvent
from .tte_date import DATETimeToEvent
from .tte_deephit import DeephitTimeToEvent
from .tte_survival_function_regression import SurvivalFunctionTimeToEvent
from .tte_tenn import TENNTimeToEvent

try:
    from .tte_xgb import XGBTimeToEvent
except ImportError:
    XGBTimeToEvent = None


def get_model_template(model: str) -> Any:
    defaults = {
        "tenn": TENNTimeToEvent,
        "date": DATETimeToEvent,
        "weibull_aft": WeibullAFTTimeToEvent,
        "cox_ph": CoxPHTimeToEvent,
        "deephit": DeephitTimeToEvent,
        "survival_function_regression": SurvivalFunctionTimeToEvent,
    }
    if XGBTimeToEvent is not None:
        defaults["survival_xgboost"] = XGBTimeToEvent

    if model in defaults:
        return defaults[model]

    raise RuntimeError(f"invalid model {model}")
