# stdlib
from typing import Any

# synthcity relative
from .ts_surv_coxph import CoxTimeSeriesSurvival
from .ts_surv_dynamic_deephit import DynamicDeephitTimeSeriesSurvival

try:
    from .ts_surv_xgb import XGBTimeSeriesSurvival
except ImportError:
    XGBTimeSeriesSurvival = None


def get_model_template(model: str) -> Any:
    defaults = {
        "cox_ph": CoxTimeSeriesSurvival,
        "dynamic_deephit": DynamicDeephitTimeSeriesSurvival,
    }
    if XGBTimeSeriesSurvival is not None:
        defaults["ts_survival_xgboost"] = XGBTimeSeriesSurvival

    if model in defaults:
        return defaults[model]

    raise RuntimeError(f"invalid model {model}")
