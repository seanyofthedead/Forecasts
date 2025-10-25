import importlib
import math
import sys
import types
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def clean_currency_functions():
    """Import clean_currency helpers with lightweight stubs for heavy deps."""

    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    created_modules = []

    def add_module(name: str, *, package: bool = False):
        module = types.ModuleType(name)
        if package:
            module.__path__ = []  # type: ignore[attr-defined]
        module.__spec__ = None  # type: ignore[assignment]
        sys.modules[name] = module
        created_modules.append(name)
        return module

    if "pandas" not in sys.modules:
        pandas = add_module("pandas")
        pandas.isna = lambda value: value != value  # type: ignore[attr-defined]

    if "numpy" not in sys.modules:
        add_module("numpy")

    if "sklearn" not in sys.modules:
        sklearn = add_module("sklearn", package=True)
        metrics = add_module("sklearn.metrics")
        metrics.mean_absolute_percentage_error = lambda y_true, y_pred: 0.0  # type: ignore[attr-defined]
        ensemble = add_module("sklearn.ensemble")
        ensemble.RandomForestRegressor = type("RandomForestRegressor", (), {})  # type: ignore[attr-defined]
        preprocessing = add_module("sklearn.preprocessing")
        preprocessing.StandardScaler = type("StandardScaler", (), {})  # type: ignore[attr-defined]

    if "scipy" not in sys.modules:
        scipy = add_module("scipy", package=True)
        stats = add_module("scipy.stats")
        scipy.stats = stats  # type: ignore[attr-defined]

    if "statsmodels" not in sys.modules:
        statsmodels = add_module("statsmodels", package=True)
        api = add_module("statsmodels.api")
        statsmodels.api = api  # type: ignore[attr-defined]

        tsa = add_module("statsmodels.tsa", package=True)
        statsmodels.tsa = tsa  # type: ignore[attr-defined]

        holtwinters = add_module("statsmodels.tsa.holtwinters")
        tsa.holtwinters = holtwinters  # type: ignore[attr-defined]
        holtwinters.ExponentialSmoothing = type("ExponentialSmoothing", (), {})  # type: ignore[attr-defined]
        holtwinters.HoltWinters = type("HoltWinters", (), {})  # type: ignore[attr-defined]

        statespace = add_module("statsmodels.tsa.statespace", package=True)
        tsa.statespace = statespace  # type: ignore[attr-defined]

        sarimax = add_module("statsmodels.tsa.statespace.sarimax")
        statespace.sarimax = sarimax  # type: ignore[attr-defined]
        sarimax.SARIMAX = type("SARIMAX", (), {})  # type: ignore[attr-defined]

    forecast_experiments = importlib.import_module("forecast_experiments")
    run_all_forecasts = importlib.import_module("run_all_forecasts")

    yield forecast_experiments.clean_currency, run_all_forecasts.clean_currency

    for name in created_modules:
        sys.modules.pop(name, None)


@pytest.mark.parametrize(
    "raw_value, expected",
    [
        ("", 0.0),
        ("-", 0.0),
        ("$1,234", 1234.0),
        ("$-", 0.0),
        ("5%", 0.05),
        ("invalid", 0.0),
    ],
)
def test_clean_currency_returns_expected_values(clean_currency_functions, raw_value, expected):
    """Ensure clean_currency from forecast_experiments handles key edge cases."""
    clean_currency_experiments, _ = clean_currency_functions
    result = clean_currency_experiments(raw_value)
    assert math.isclose(result, expected, rel_tol=1e-9, abs_tol=0.0)


@pytest.mark.parametrize(
    "raw_value",
    ["", "-", "$1,234", "$-", "5%", "invalid"],
)
def test_clean_currency_consistency_across_modules(clean_currency_functions, raw_value):
    """Verify the helper behaves consistently in both scripts."""
    clean_currency_experiments, clean_currency_run_all = clean_currency_functions
    result_experiments = clean_currency_experiments(raw_value)
    result_run_all = clean_currency_run_all(raw_value)
    assert result_experiments == result_run_all
