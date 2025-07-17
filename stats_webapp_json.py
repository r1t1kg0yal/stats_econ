#!/usr/bin/env python3
"""
panel_json_builder.py

Builds and returns the JSON representation of the **Time Series Scatter & Diagnostics**
panel shown in the Dash app (see screenshot).  
This version adds **per‑chart annotations** to the JSON, one entry for every
diagnostic chart that appears in the panel.

All helper functions remain self‑contained so the file can be dropped into an
existing code‑base with minimal friction.
"""

import json
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import statsmodels.api as sm
from dash import dcc
from dash.dependencies import Input, Output, State
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.tsa.stattools import (adfuller, grangercausalitytests, kpss)
from statsmodels.tsa.vector_ar.vecm import coint_johansen

# Your local helpers (must be import‑able in the same environment)
import stats_webapp_helper as helper


# --------------------------------------------------------------------------------------
# Utility helpers
# --------------------------------------------------------------------------------------
def _to_python_scalars(obj: Any) -> Any:
    """
    Recursively convert NumPy scalars to native Python scalars so the structure
    can be serialized cleanly with `json.dumps`.
    """
    if isinstance(obj, dict):
        return {k: _to_python_scalars(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [_to_python_scalars(v) for v in obj]

    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)

    if isinstance(obj, (np.floating, np.float64)):
        return float(obj)

    if isinstance(obj, (np.bool_,)):
        return bool(obj)

    return obj


def apply_transforms_to_series(
    series: pd.Series,
    transforms: List[str],
    data_frame: pd.DataFrame,
    period_map: Dict[str, Dict[str, int]],
    native_frequency: str,
) -> pd.Series:
    """
    Apply the list of transformation strings (YoY % Change, Log, Inflation Adjust, …)
    to the input series, obeying the period map so lag lengths are automatically
    aligned to the series’ native frequency.
    """
    txs = list(transforms or [])
    result = series.copy()

    # Inflation adjustment (always first so downstream diffs/pct_change operate on
    # real values if the user selected this option)
    if 'Inflation Adjust' in txs:
        cpi = data_frame['USA CPI'].reindex(result.index)
        result = result.divide(cpi, axis=0)
        txs.remove('Inflation Adjust')

    # Other transforms in the order they were chosen
    for t in txs:
        if t.startswith('YoY'):
            lag = period_map['YoY'][native_frequency]
            result = result.pct_change(lag) * 100 if '% Change' in t else result.diff(lag)
        elif t.startswith('QoQ'):
            lag = period_map['QoQ'][native_frequency]
            result = result.pct_change(lag) * 100 if '% Change' in t else result.diff(lag)
        elif t.startswith('MoM'):
            lag = period_map['MoM'][native_frequency]
            result = result.pct_change(lag) * 100 if '% Change' in t else result.diff(lag)
        elif t == 'Log':
            result = np.log10(result.replace(0, np.nan))

    return result


def apply_lag_to_series(series: pd.Series, lag_periods: int) -> pd.Series:
    """Shift the series forward by *lag_periods* (positive integer)."""
    return series.shift(lag_periods)


# --------------------------------------------------------------------------------------
# Core JSON builder
# --------------------------------------------------------------------------------------
def get_diag_dict(
    data_frame: pd.DataFrame,
    frequency_to_rule: Dict[str, str],
    period_map: Dict[str, Dict[str, int]],
    native_frequency: str,
    x_variable: str,
    x_frequency: str,
    x_transforms: List[str],
    x_lag: int,
    y_variable: str,
    y_frequency: str,
    y_transforms: List[str],
    y_lag: int,
    fit_type: str,
    dual_axis_flags: List[str],
    diagnostic_choices: List[str],
) -> Dict[str, Any]:
    """
    Assemble every statistic that populates the panel **and** attach a list of
    `{ chart, annotation }` dicts for all diagnostic charts.
    """

    # ------------------------------------------------------------------
    # Build transformed / lagged series and common combined DataFrame
    # ------------------------------------------------------------------
    base_x_series = apply_transforms_to_series(
        data_frame[x_variable], x_transforms, data_frame, period_map, native_frequency
    )
    base_y_series = apply_transforms_to_series(
        data_frame[y_variable], y_transforms, data_frame, period_map, native_frequency
    )

    transformed_x = apply_lag_to_series(base_x_series, x_lag)
    transformed_y = apply_lag_to_series(base_y_series, y_lag)

    series_x = transformed_x.resample(frequency_to_rule[x_frequency]).last().dropna()
    series_y = transformed_y.resample(frequency_to_rule[y_frequency]).last().dropna()

    combined_df = pd.concat(
        [series_x.rename(x_variable), series_y.rename(y_variable)], axis=1
    ).dropna()

    # ------------------------------------------------------------------
    # Core regression and goodness‑of‑fit metrics
    # ------------------------------------------------------------------
    design_matrix = sm.add_constant(combined_df[x_variable])
    regression_model = sm.OLS(combined_df[y_variable], design_matrix).fit()

    # Scatter‑plot trend‑line parameters
    if fit_type == 'linear':
        slope, intercept = np.polyfit(combined_df[x_variable], combined_df[y_variable], 1)
        predictions = slope * combined_df[x_variable] + intercept
    elif fit_type == 'exp':
        mask = combined_df[y_variable] > 0
        log_slope, log_int = np.polyfit(
            combined_df.loc[mask, x_variable],
            np.log(combined_df.loc[mask, y_variable]),
            1,
        )
        a = np.exp(log_int)
        predictions = a * np.exp(log_slope * combined_df[x_variable])
    else:  # quadratic
        a2, b2, c2 = np.polyfit(combined_df[x_variable], combined_df[y_variable], 2)
        predictions = a2 * combined_df[x_variable] ** 2 + b2 * combined_df[x_variable] + c2

    r_squared = 1 - np.sum((combined_df[y_variable] - predictions) ** 2) / np.sum(
        (combined_df[y_variable] - combined_df[y_variable].mean()) ** 2
    )

    slope_coef = float(regression_model.params[x_variable])
    p_val_slope = float(regression_model.pvalues[x_variable])

    # ------------------------------------------------------------------
    # Regression diagnostics
    # ------------------------------------------------------------------
    dw_stat: float = float(durbin_watson(regression_model.resid))
    bp_stat, bp_p, _, _ = het_breuschpagan(regression_model.resid, design_matrix)
    jb_stat, jb_p, _, _ = jarque_bera(regression_model.resid)

    # ------------------------------------------------------------------
    # Granger causality
    # ------------------------------------------------------------------
    gc_df = pd.concat(
        [series_y.rename(y_variable), series_x.rename(x_variable)], axis=1
    ).dropna()
    max_lag = min(4, len(gc_df) - 1)
    gc_results = grangercausalitytests(gc_df, maxlag=max_lag, verbose=False) if max_lag >= 1 else {}

    # ------------------------------------------------------------------
    # Unit‑root tests
    # ------------------------------------------------------------------
    adf_x_s, adf_x_p = adfuller(series_x, autolag='AIC')[:2]
    kpss_x_s, kpss_x_p = kpss(series_x, regression='c', nlags='auto')[:2]
    adf_y_s, adf_y_p = adfuller(series_y, autolag='AIC')[:2]
    kpss_y_s, kpss_y_p = kpss(series_y, regression='c', nlags='auto')[:2]

    unit_root_results = [
        {
            "variable": "X",
            "test": "ADF",
            "stat": adf_x_s,
            "p_value": adf_x_p,
            "stationary": adf_x_p < 0.05,
        },
        {
            "variable": "X",
            "test": "KPSS",
            "stat": kpss_x_s,
            "p_value": kpss_x_p,
            "stationary": kpss_x_p >= 0.05,
        },
        {
            "variable": "Y",
            "test": "ADF",
            "stat": adf_y_s,
            "p_value": adf_y_p,
            "stationary": adf_y_p < 0.05,
        },
        {
            "variable": "Y",
            "test": "KPSS",
            "stat": kpss_y_s,
            "p_value": kpss_y_p,
            "stationary": kpss_y_p >= 0.05,
        },
    ]

    # ------------------------------------------------------------------
    # Cointegration tests
    # ------------------------------------------------------------------
    ci_df = pd.concat(
        [series_x.rename(x_variable), series_y.rename(y_variable)], axis=1
    ).dropna()

    res_eg = sm.OLS(ci_df[y_variable], sm.add_constant(ci_df[x_variable])).fit().resid
    eg_stat, eg_p = adfuller(res_eg, autolag='AIC')[:2]

    joh = coint_johansen(ci_df, det_order=0, k_ar_diff=1)
    joh_trace, joh_crit95 = float(joh.lr1[0]), float(joh.cvt[0, 1])

    cointegration_results = {
        "engle_granger": {
            "stat": eg_stat,
            "p_value": eg_p,
            "cointegrated": eg_p < 0.05,
        },
        "johansen_trace": {
            "stat": joh_trace,
            "crit_95": joh_crit95,
            "cointegrated": joh_trace > joh_crit95,
        },
    }

    # ------------------------------------------------------------------
    # Chart annotations – NEW (actual verdicts)
    # ------------------------------------------------------------------
    diag_annots = {}
    for name in diagnostic_choices or []:
        if name == 'R² vs Lag':
            base_df = pd.concat(
                [
                    series_x.rename(x_variable),
                    series_y.rename(y_variable),
                ],
                axis=1
            )
            _, annot = helper.r2_vs_lag(
                base_df,
                x_variable,
                y_variable,
                period_map['YoY'][x_frequency]
            )
        elif name == 'Residuals vs Fitted':
            _, annot = helper.residuals_vs_fitted(regression_model)
        elif name == 'Residuals Histogram':
            _, annot = helper.residuals_histogram(regression_model.resid)
        elif name == 'Normal Q–Q Plot':
            _, annot = helper.normal_qq_plot(regression_model.resid)
        elif name == 'Residuals ACF':
            _, annot = helper.residuals_acf(regression_model.resid)
        elif name == 'Residuals PACF':
            _, annot = helper.residuals_pacf(regression_model.resid)
        elif name == 'Scale–Location Plot':
            _, annot = helper.scale_location_plot(regression_model)
        elif name == 'Influence Plot':
            _, annot = helper.influence_plot(regression_model)
        elif name == 'CUSUM of Residuals':
            _, annot = helper.cusum_plot(regression_model.resid)
        else:
            annot = ""
        diag_annots[name] = annot
    chart_annotations = [
        {"chart": chart_name, "annotation": diag_annots.get(chart_name, "")}
        for chart_name in diagnostic_choices
    ]

    # ------------------------------------------------------------------
    # Final dictionary to return
    # ------------------------------------------------------------------
    # Durbin-Watson interpretation
    if dw_stat >= 1.8 and dw_stat <= 2.2:
        dw_interp = "No Autocorrelation"
    elif dw_stat < 1.8:
        dw_interp = "Positive Autocorrelation"
    else:
        dw_interp = "Negative Autocorrelation"

    # Regression interpretation
    regression_interp = "Significant" if p_val_slope < 0.05 else "Not significant"

    diag_dict: Dict[str, Any] = {
        "x_var": x_variable,
        "y_var": y_variable,
        "x_freq": x_frequency,
        "y_freq": y_frequency,
        "x_transforms": x_transforms,
        "y_transforms": y_transforms,
        "x_lag_periods": x_lag,
        "y_lag_periods": y_lag,
        "observations": {"x": int(len(series_x)), "y": int(len(series_y))},
        "regression": {
            "slope_coef": slope_coef,
            "p_value": p_val_slope,
            "r_squared": r_squared,
            "regression_type": fit_type,
            "regression_interpret": regression_interp,
        },
        "diagnostics": {
            "durbin_watson": dw_stat,
            "durbin_watson_interpret": dw_interp,
            "breusch_pagan": {
                "stat": bp_stat,
                "p_value": bp_p,
                "interpret": "heteroskedastic" if bp_p < 0.05 else "homoskedastic",
            },
            "jarque_bera": {
                "stat": jb_stat,
                "p_value": jb_p,
                "interpret": "non‑normal residuals" if jb_p < 0.05 else "normal residuals",
            },
        },
        "granger": [
            {
                "lag": lag,
                "p_value": float(gc_results[lag][0]['ssr_ftest'][1]),
                "interpret": "reject H₀" if gc_results[lag][0]['ssr_ftest'][1] < 0.05 else "fail to reject",
            }
            for lag in range(1, max_lag + 1)
        ] if gc_results else [],
        "unit_root": unit_root_results,
        "cointegration": cointegration_results,
        "chart_annotations": chart_annotations,   # << NEW >>
    }

    return diag_dict


# --------------------------------------------------------------------------------------
# Dash callback registration
# --------------------------------------------------------------------------------------
def register_json_download_callback(
    app,
    data_frame: pd.DataFrame,
    frequency_to_rule: Dict[str, str],
    period_map: Dict[str, Dict[str, int]],
    native_frequency: str,
) -> None:
    """
    Attach a Dash callback that generates and **downloads** the JSON file whenever
    the user clicks *Download Panel as JSON*.
    """

    def _generate_diag_dict(
        x_variable, x_frequency, x_transforms, x_lag,
        y_variable, y_frequency, y_transforms, y_lag,
        fit_type, dual_axis_flags, diagnostic_choices
    ):
        return get_diag_dict(
            data_frame,
            frequency_to_rule,
            period_map,
            native_frequency,
            x_variable,
            x_frequency,
            x_transforms,
            x_lag,
            y_variable,
            y_frequency,
            y_transforms,
            y_lag,
            fit_type,
            dual_axis_flags,
            diagnostic_choices,
        )

    # Callback ---------------------------------------------------------
    @app.callback(
        Output('download-json', 'data'),
        Input('download-json-btn', 'n_clicks'),
        State('scatter-fit-dropdown', 'value'),
        State('x-axis-dropdown', 'value'),
        State('x-freq-dropdown', 'value'),
        State('x-transform-dropdown', 'value'),
        State('x-lag-periods', 'value'),
        State('y-axis-dropdown', 'value'),
        State('y-freq-dropdown', 'value'),
        State('y-transform-dropdown', 'value'),
        State('y-lag-periods', 'value'),
        State('dual-axis-checkbox', 'value'),
        prevent_initial_call=True,
    )
    def download_panel_json(
        n_clicks,
        fit_type,
        x_var, x_freq, x_tx, x_lag,
        y_var, y_freq, y_tx, y_lag,
        dual_flags,
    ):
        # Always include **all** diagnostic charts except Rolling R²
        diag_choices = [
            name for name in helper.chart_functions.keys() if name != 'Rolling R²'
        ]

        diag_dict = _generate_diag_dict(
            x_var, x_freq, x_tx, x_lag,
            y_var, y_freq, y_tx, y_lag,
            fit_type, dual_flags, diag_choices,
        )

        # Clean non‑serializable NumPy objects
        diag_dict = _to_python_scalars(diag_dict)

        filename = f"{x_var.replace(' ', '_')}_vs_{y_var.replace(' ', '_')}.json"
        return dcc.send_string(json.dumps(diag_dict, indent=2), filename=filename)
