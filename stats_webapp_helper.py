#!/usr/bin/env python3
"""
stats_webapp_helper.py

Reusable diagnostic‑chart functions for the Dash app.
Each chart (except R²‑vs‑Lag) is now automatically annotated
with a simple binary verdict: something like
    ✓ Roughly Normal
or
    ✗ Not Normal
to flag whether the diagnostic looks “good” or “bad”.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from statsmodels.tsa.stattools import acf, pacf

# ----------------------------------------------------------------------
# Thresholds for good vs bad outcomes (edit here to change globally)
# ----------------------------------------------------------------------
RESIDUALS_VS_FITTED_SLOPE_THRESHOLD = 0.05  # as fraction of |resid| range
NORMAL_QQ_R2_THRESHOLD = 0.9
ACF_BARTLETT_BAND = 2.0  # multiplier for 1/sqrt(n)
PACF_BARTLETT_BAND = 2.0
SCALE_LOCATION_SLOPE_THRESHOLD = 0.05
INFLUENCE_COOKS_THRESHOLD_FACTOR = 4  # threshold = 4/n
CUSUM_SIGMA_THRESHOLD = 1.96
RESIDUALS_HISTOGRAM_P_THRESHOLD = 0.01

# ----------------------------------------------------------------------
# 0.  Utility helpers
# ----------------------------------------------------------------------
def add_annotation(fig: go.Figure, verdict: str, good: bool) -> None:
    """Place a green (good) or red (bad) label in the top‑left corner."""
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.02,
        y=0.98,
        text=("✓ " if good else "✗ ") + verdict,
        showarrow=False,
        font=dict(color="green" if good else "red", size=12),
        bgcolor="rgba(255,255,255,0.7)",
        bordercolor="rgba(0,0,0,0.3)",
        borderwidth=1,
        borderpad=4,
    )


# ----------------------------------------------------------------------
# 1.  R² vs lag  – (no annotation requested)
# ----------------------------------------------------------------------
def r2_vs_lag(
    combined_df: pd.DataFrame,
    x_var: str,
    y_var: str,
    periods_per_unit: int,
    years: int = 20,
) -> go.Figure:
    """Plot R² of regressing Y on X shifted by successive lags."""
    max_lag = periods_per_unit * years
    lags = list(range(max_lag + 1))
    r2_values: list[float] = []

    for lag in lags:
        df = combined_df.copy()
        df[x_var] = df[x_var].shift(lag)
        df.dropna(inplace=True)

        if len(df) > 1:
            slope, intercept = np.polyfit(df[x_var], df[y_var], 1)
            pred = slope * df[x_var] + intercept
            rss = np.sum((df[y_var] - pred) ** 2)
            tss = np.sum((df[y_var] - df[y_var].mean()) ** 2)
            r2 = 1 - rss / tss
        else:
            r2 = np.nan
        r2_values.append(r2)

    fig = px.line(
        x=lags,
        y=r2_values,
        labels={"x": "Lag (periods)", "y": "R²"},
        title="R² vs Lag",
    )
    fig.update_layout(title_x=0.5, margin={"t": 30})
    # Add annotation for best lag
    if any(np.isfinite(r2_values)):
        best_lag = int(np.nanargmax(r2_values))
        best_r2 = r2_values[best_lag]
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.98, y=0.98,
            text=f"Best Lag: {best_lag}",
            showarrow=False,
            font=dict(color="blue", size=12),
            bgcolor="rgba(255,255,255,0.7)",
            bordercolor="rgba(0,0,0,0.3)",
            borderwidth=1,
            borderpad=4,
            xanchor="right",
            yanchor="top"
        )
    return fig, ""


# ----------------------------------------------------------------------
# 2.  Residual‑based diagnostics
# ----------------------------------------------------------------------
def residuals_vs_fitted(reg_model) -> tuple[go.Figure, str]:
    """Residuals vs fitted values with ‘no‑pattern’ verdict."""
    fitted = reg_model.fittedvalues
    resid = reg_model.resid

    fig = px.scatter(
        x=fitted,
        y=resid,
        labels={"x": "Fitted Values", "y": "Residuals"},
        title="Residuals vs Fitted",
    )

    # Fit a straight line to detect any trend
    if len(fitted) > 1:
        slope, intercept = np.polyfit(fitted, resid, 1)
        xx = np.linspace(fitted.min(), fitted.max(), 200)
        fig.add_trace(
            go.Scatter(
                x=xx,
                y=slope * xx + intercept,
                mode="lines",
                line=dict(dash="dash"),
                showlegend=False,
            )
        )
    else:
        slope = 0.0

    # Heuristic: if |slope| is tiny (<.05*|resid| range) treat as “good”
    rng = resid.max() - resid.min() if len(resid) else 1
    good = abs(slope) < RESIDUALS_VS_FITTED_SLOPE_THRESHOLD * max(rng, 1e-9)
    verdict = "No obvious pattern" if good else "Pattern detected"
    add_annotation(fig, verdict, good)
    fig.update_layout(title_x=0.5, margin={"t": 30})
    return fig, verdict


def residuals_histogram(residuals: pd.Series) -> tuple[go.Figure, str]:
    """Histogram of residuals with Normal‑vs‑non‑Normal verdict."""
    fig = px.histogram(
        residuals,
        nbins=30,
        labels={"value": "Residuals"},
        title="Residuals Histogram",
    )
    fig.update_layout(title_x=0.5, margin={"t": 30}, showlegend=False)

    # D'Agostino–Pearson normality test
    if residuals.dropna().size >= 8:  # test requires n≥8
        k2, p = stats.normaltest(residuals.dropna())
        good = p >= RESIDUALS_HISTOGRAM_P_THRESHOLD
    else:
        good = False  # not enough data

    verdict = "Roughly Normal" if good else "Not Normal"
    add_annotation(fig, verdict, good)
    return fig, verdict


def normal_qq_plot(residuals: pd.Series) -> tuple[go.Figure, str]:
    """Normal Q–Q plot annotated with Normal vs not."""
    sorted_resid = np.sort(residuals)
    theor_q = stats.norm.ppf((np.arange(len(sorted_resid)) + 0.5) / len(sorted_resid))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=theor_q, y=sorted_resid, mode="markers", showlegend=False)
    )
    slope, intercept, r, _, _ = stats.linregress(theor_q, sorted_resid)
    x0, x1 = theor_q.min(), theor_q.max()
    fig.add_trace(
        go.Scatter(
            x=[x0, x1],
            y=[slope * x0 + intercept, slope * x1 + intercept],
            mode="lines",
            line=dict(dash="dash"),
            showlegend=False,
        )
    )
    fig.update_layout(
        xaxis_title="Theoretical Quantiles",
        yaxis_title="Ordered Residuals",
        title="Normal Q–Q Plot",
        title_x=0.5,
        margin={"t": 30},
    )

    good = r**2 >= NORMAL_QQ_R2_THRESHOLD  # correlation very close to 1 ⇒ straight line
    verdict = "Roughly Normal" if good else "Not Normal"
    add_annotation(fig, verdict, good)
    return fig, verdict


def residuals_acf(residuals: pd.Series, nlags: int = 40) -> tuple[go.Figure, str]:
    """ACF bar chart with ‘autocorr / no autocorr’ verdict."""
    acf_vals = acf(residuals, nlags=nlags)
    lags = range(len(acf_vals))

    fig = px.bar(
        x=lags,
        y=acf_vals,
        labels={"x": "Lag", "y": "ACF"},
        title="Residuals ACF",
    )
    fig.update_layout(title_x=0.5, margin={"t": 30})

    # 95 % Bartlett band ≈ ±2/√n
    n = residuals.dropna().size
    crit = ACF_BARTLETT_BAND / np.sqrt(max(n, 1))
    good = not np.any(np.abs(acf_vals[1:]) > crit)  # ignore lag0

    verdict = "No autocorrelation" if good else "Autocorrelation"
    add_annotation(fig, verdict, good)
    return fig, verdict


def residuals_pacf(residuals: pd.Series, nlags: int = 40) -> tuple[go.Figure, str]:
    """PACF bar chart with verdict."""
    # Ensure nlags does not exceed half the sample size minus one
    max_nlags = (len(residuals.dropna()) // 2) - 1
    nlags = min(nlags, max_nlags) if max_nlags > 0 else 1
    pacf_vals = pacf(residuals, nlags=nlags, method="yw")
    lags = range(len(pacf_vals))

    fig = px.bar(
        x=lags,
        y=pacf_vals,
        labels={"x": "Lag", "y": "PACF"},
        title="Residuals PACF",
    )
    fig.update_layout(title_x=0.5, margin={"t": 30})

    n = residuals.dropna().size
    crit = PACF_BARTLETT_BAND / np.sqrt(max(n, 1))
    good = not np.any(np.abs(pacf_vals[1:]) > crit)

    verdict = "No partial autocorr" if good else "Partial autocorr"
    add_annotation(fig, verdict, good)
    return fig, verdict


def scale_location_plot(reg_model) -> tuple[go.Figure, str]:
    """√|standardised residuals| vs fitted with homosked‑vs‑heterosked verdict."""
    fitted = reg_model.fittedvalues
    resid = reg_model.resid
    std_resid = resid / np.std(resid)
    sqrt_std = np.sqrt(np.abs(std_resid))

    fig = px.scatter(
        x=fitted,
        y=sqrt_std,
        labels={"x": "Fitted Values", "y": "√|Standardised Residuals|"},
        title="Scale–Location Plot",
    )
    fig.update_layout(title_x=0.5, margin={"t": 30})

    # Slope near zero means variance roughly constant
    if len(fitted) > 1:
        slope, _ = np.polyfit(fitted, sqrt_std, 1)
    else:
        slope = 0.0
    good = abs(slope) < SCALE_LOCATION_SLOPE_THRESHOLD

    verdict = "Homoskedastic" if good else "Heteroskedastic"
    add_annotation(fig, verdict, good)
    return fig, verdict


def influence_plot(reg_model) -> tuple[go.Figure, str]:
    """Leverage vs studentised residuals, verdict: influential points or not."""
    infl = reg_model.get_influence()
    leverage = infl.hat_matrix_diag
    studentized = infl.resid_studentized_internal
    cooks = infl.cooks_distance[0]

    fig = px.scatter(
        x=leverage,
        y=studentized,
        size=cooks,
        labels={"x": "Leverage", "y": "Studentised Residuals"},
        title="Influence Plot",
    )
    fig.update_layout(title_x=0.5, margin={"t": 30})

    n = len(leverage)
    threshold = INFLUENCE_COOKS_THRESHOLD_FACTOR / n
    num_big = int(np.sum(cooks > threshold))
    good = num_big == 0

    verdict = "No large influences" if good else f"{num_big} influential point(s)"
    add_annotation(
        fig,
        verdict,
        good,
    )
    return fig, verdict


def cusum_plot(residuals: pd.Series) -> tuple[go.Figure, str]:
    """CUSUM chart with stability verdict."""
    cumsum = residuals.cumsum()
    fig = px.line(
        x=residuals.index,
        y=cumsum,
        labels={"x": "Date", "y": "Cumulative Residuals"},
        title="CUSUM of Residuals",
    )
    fig.update_layout(title_x=0.5, margin={"t": 30})

    # Simple rule: final |cusum| within 1.96·σ ⇒ stable
    sigma = np.std(residuals)
    good = abs(cumsum.iloc[-1]) <= CUSUM_SIGMA_THRESHOLD * sigma

    verdict = "Stable" if good else "Structural shift?"
    add_annotation(fig, verdict, good)
    return fig, verdict


# ----------------------------------------------------------------------
# 3.  Convenience mapping for the Dash dropdown
# ----------------------------------------------------------------------
chart_functions = {
    "R² vs Lag": r2_vs_lag,
    "Residuals vs Fitted": residuals_vs_fitted,
    "Residuals Histogram": residuals_histogram,
    "Normal Q–Q Plot": normal_qq_plot,
    "Residuals ACF": residuals_acf,
    "Residuals PACF": residuals_pacf,
    "Scale–Location Plot": scale_location_plot,
    "Influence Plot": influence_plot,
    "CUSUM of Residuals": cusum_plot,
    "Rolling R²": None,  # kept for completeness; not used in dropdown
}

# Plain-text chart annotations for JSON export (no check/X)
chart_annotations = {
    "R² vs Lag": "",  # No annotation requested
    "Residuals vs Fitted": "No obvious pattern if slope is small, Pattern detected otherwise",
    "Residuals Histogram": "Roughly Normal if p >= 0.01, Not Normal otherwise",
    "Normal Q–Q Plot": "Roughly Normal if R² >= 0.9, Not Normal otherwise",
    "Residuals ACF": "No autocorrelation if all lags within ±2/sqrt(n), Autocorrelation otherwise",
    "Residuals PACF": "No partial autocorr if all lags within ±2/sqrt(n), Partial autocorr otherwise",
    "Scale–Location Plot": "Homoskedastic if slope near zero, Heteroskedastic otherwise",
    "Influence Plot": "No large influences if no Cook's D > 4/n, else number of influential points",
    "CUSUM of Residuals": "Stable if final |cusum| <= 1.96·σ, Structural shift otherwise"
}
