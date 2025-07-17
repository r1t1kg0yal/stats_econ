import pytest
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from statsmodels.regression.linear_model import OLS
import stats_webapp_helper as helper
import stats_webapp_json as swj

# python3 -m pytest test_stats_webapp.py

# --- Fixtures ---
@pytest.fixture
def simple_series():
    return pd.Series([1, 2, 3, 4, 5])

@pytest.fixture
def normal_resid():
    np.random.seed(0)
    return pd.Series(np.random.normal(0, 1, 100))

@pytest.fixture
def heteroskedastic_resid():
    np.random.seed(0)
    return pd.Series(np.random.normal(0, np.linspace(1, 3, 100)))

@pytest.fixture
def reg_model():
    x = np.arange(100)
    y = 2 * x + np.random.normal(0, 1, 100)
    df = pd.DataFrame({'x': x, 'y': y})
    model = OLS(df['y'], sm.add_constant(df['x'])).fit()
    return model

# --- Diagnostic chart function tests ---
def test_residuals_vs_fitted_returns_annotation(reg_model):
    fig, annot = helper.residuals_vs_fitted(reg_model)
    assert isinstance(fig, go.Figure)
    assert annot in ["No obvious pattern", "Pattern detected"]

def test_residuals_histogram_annotation(normal_resid):
    fig, annot = helper.residuals_histogram(normal_resid)
    assert isinstance(fig, go.Figure)
    assert annot in ["Roughly Normal", "Not Normal"]

def test_normal_qq_plot_annotation(normal_resid):
    fig, annot = helper.normal_qq_plot(normal_resid)
    assert isinstance(fig, go.Figure)
    assert annot in ["Roughly Normal", "Not Normal"]

def test_residuals_acf_annotation(normal_resid):
    fig, annot = helper.residuals_acf(normal_resid)
    assert isinstance(fig, go.Figure)
    assert annot in ["No autocorrelation", "Autocorrelation"]

def test_residuals_pacf_annotation(normal_resid):
    fig, annot = helper.residuals_pacf(normal_resid)
    assert isinstance(fig, go.Figure)
    assert annot in ["No partial autocorr", "Partial autocorr"]

def test_scale_location_plot_annotation(reg_model):
    fig, annot = helper.scale_location_plot(reg_model)
    assert isinstance(fig, go.Figure)
    assert annot in ["Homoskedastic", "Heteroskedastic"]

def test_influence_plot_annotation(reg_model):
    fig, annot = helper.influence_plot(reg_model)
    assert isinstance(fig, go.Figure)
    assert "influential point" in annot or annot == "No large influences"

def test_cusum_plot_annotation(normal_resid):
    fig, annot = helper.cusum_plot(normal_resid)
    assert isinstance(fig, go.Figure)
    assert annot in ["Stable", "Structural shift?"]

# --- JSON export logic ---
def test_json_chart_annotations_match_helper(monkeypatch):
    # Minimal DataFrame and config
    df = pd.DataFrame({
        'Date': pd.date_range('2000-01-01', periods=100, freq='D'),
        'X': np.random.normal(0, 1, 100),
        'Y': np.random.normal(0, 1, 100),
        'USA CPI': np.ones(100)
    }).set_index('Date')
    freq_rule = {'daily': 'D'}
    period_map = {'YoY': {'daily': 365}, 'QoQ': {'daily': 90}, 'MoM': {'daily': 30}}
    native_freq = 'daily'
    diag_choices = [k for k in helper.chart_functions.keys() if k != 'Rolling R²']
    result = swj.get_diag_dict(
        df, freq_rule, period_map, native_freq,
        'X', 'daily', [], 0, 'Y', 'daily', [], 0, 'linear', [], diag_choices
    )
    # Check that chart_annotations are present and non-empty
    assert 'chart_annotations' in result
    for entry in result['chart_annotations']:
        assert 'chart' in entry and 'annotation' in entry
        assert entry['annotation'] != ""

# --- Edge cases ---
def test_short_series_pacf():
    short = pd.Series([1, 2, 3, 4])
    fig, annot = helper.residuals_pacf(short)
    assert isinstance(fig, go.Figure)
    assert annot in ["No partial autocorr", "Partial autocorr"]

def test_all_nan_series():
    nan_series = pd.Series([np.nan, np.nan, np.nan, np.nan])
    fig, annot = helper.residuals_histogram(nan_series)
    assert isinstance(fig, go.Figure)
    assert annot in ["Roughly Normal", "Not Normal"] 