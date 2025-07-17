#!/usr/bin/env python3
"""
time_series_scatter_and_diagnostics.py

Dash app for interactive scatter / time‑series plots and full statistical
diagnostics, now with a “Download Panel as JSON” button.
"""

# ---------------------------------------------------------------------
# 0. Imports
# ---------------------------------------------------------------------
import json
import numpy as np
import pandas as pd
import io
import base64

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go

import statsmodels.api as sm
from statsmodels.stats.stattools import jarque_bera, durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests
from statsmodels.tsa.vector_ar.vecm import coint_johansen

import stats_webapp_helper as helper  # your existing helper module
from stats_webapp_json import register_json_download_callback
# ---------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------

frequency_to_rule = {
    'daily':     'D',
    'weekly':    'W',
    'monthly':   'M',
    'quarterly': 'Q',
    'annual':    'A'
}
frequency_to_unit = {
    'daily':     'Days',
    'weekly':    'Weeks',
    'monthly':   'Months',
    'quarterly': 'Quarters',
    'annual':    'Years'
}
_period_map = {               # dynamic lag sizes for each native frequency
    'YoY': {'daily': 365, 'weekly': 52,  'monthly': 12, 'quarterly': 4, 'annual': 1},
    'QoQ': {'daily':  90, 'weekly': 13,  'monthly':  3, 'quarterly': 1},
    'MoM': {'daily':  30, 'weekly':  4,  'monthly':  1}
}
# ---------------------------------------------------------------------
# 2. Remove initial data loading and setup
# ---------------------------------------------------------------------
# data_frame = pd.read_excel('data_quarterly.xlsx', parse_dates=['Date'])
# data_frame.set_index('Date', inplace=True)
# date_diffs = data_frame.index.to_series().diff().dt.days.dropna()
# median_days = date_diffs.median()
# if median_days <= 1:
#     native_frequency = 'daily'
# elif median_days <= 7:
#     native_frequency = 'weekly'
# elif median_days <= 31:
#     native_frequency = 'monthly'
# elif median_days <= 92:
#     native_frequency = 'quarterly'
# else:
#     native_frequency = 'annual'
# allowed_frequencies_by_native = {
#     'daily':     ['daily','weekly','monthly','quarterly','annual'],
#     'weekly':    ['weekly','monthly','quarterly','annual'],
#     'monthly':   ['monthly','quarterly','annual'],
#     'quarterly': ['quarterly','annual'],
#     'annual':    ['annual']
# }
# allowed_frequencies = allowed_frequencies_by_native[native_frequency]
allowed_frequencies_by_native = {
    'daily':     ['daily','weekly','monthly','quarterly','annual'],
    'weekly':    ['weekly','monthly','quarterly','annual'],
    'monthly':   ['monthly','quarterly','annual'],
    'quarterly': ['quarterly','annual'],
    'annual':    ['annual']
}
# ---------------------------------------------------------------------
# 3. Transformation options
# ---------------------------------------------------------------------
inflation_adjust_option = {'label': 'Inflation Adjust', 'value': 'Inflation Adjust'}
other_transform_options = [
    {'label': 'Year‑over‑Year % Change',             'value': 'YoY % Change'},
    {'label': 'Year‑over‑Year Arithmetic Change',    'value': 'YoY Arithmetic Change'},
    {'label': 'Quarter‑over‑Quarter % Change',       'value': 'QoQ % Change'},
    {'label': 'Quarter‑over‑Quarter Arithmetic Change','value': 'QoQ Arithmetic Change'},
    {'label': 'Month‑over‑Month % Change',           'value': 'MoM % Change'},
    {'label': 'Month‑over‑Month Arithmetic Change',  'value': 'MoM Arithmetic Change'},
    {'label': 'Log (base 10)',                       'value': 'Log'}
]
allowed_transformations_by_frequency = {
    'daily':     ['YoY % Change','YoY Arithmetic Change','QoQ % Change','QoQ Arithmetic Change',
                  'MoM % Change','MoM Arithmetic Change','Log'],
    'weekly':    ['YoY % Change','YoY Arithmetic Change','QoQ % Change','QoQ Arithmetic Change',
                  'MoM % Change','MoM Arithmetic Change','Log'],
    'monthly':   ['YoY % Change','YoY Arithmetic Change','QoQ % Change','QoQ Arithmetic Change',
                  'MoM % Change','MoM Arithmetic Change','Log'],
    'quarterly': ['YoY % Change','YoY Arithmetic Change','QoQ % Change','QoQ Arithmetic Change','Log'],
    'annual':    ['YoY % Change','YoY Arithmetic Change','Log']
}
# transform_options = [inflation_adjust_option] + [
#     opt for opt in other_transform_options
#     if opt['value'] in allowed_transformations_by_frequency[native_frequency]
# ]
# ---------------------------------------------------------------------
# 4. Variable whitelist (prefixes)
# ---------------------------------------------------------------------
allowed_prefixes = ('USA','EUR','JPN','SWI','DEU','FRA','ESP','ITA','CAN','AUS','WLD')
# allowed_variables = [c for c in data_frame.columns if c.startswith(allowed_prefixes)]
# ---------------------------------------------------------------------
# 5. Helper functions
# ---------------------------------------------------------------------
def apply_transforms_to_series(series: pd.Series, transforms: list, native_frequency: str) -> pd.Series:
    """Apply selected transformations (inflation adjustment + various diffs/logs)."""
    txs = list(transforms or [])
    result = series.copy()

    # Inflation adjust first
    if 'Inflation Adjust' in txs:
        # Defensive: check for CPI column
        if 'USA CPI' in series.index or 'USA CPI' in series:
            cpi = series['USA CPI'].reindex(result.index)
        else:
            cpi = 1
        result = result.divide(cpi, axis=0)
        txs.remove('Inflation Adjust')

    # Other transforms
    for t in txs:
        if t.startswith('YoY'):
            lag = _period_map['YoY'][native_frequency]
            result = (result.pct_change(lag)*100 if '% Change' in t else result.diff(lag))
        elif t.startswith('QoQ'):
            lag = _period_map['QoQ'][native_frequency]
            result = (result.pct_change(lag)*100 if '% Change' in t else result.diff(lag))
        elif t.startswith('MoM'):
            lag = _period_map['MoM'][native_frequency]
            result = (result.pct_change(lag)*100 if '% Change' in t else result.diff(lag))
        elif t == 'Log':
            result = np.log10(result.replace(0, np.nan))

    return result

def apply_lag_to_series(series: pd.Series, lag_periods: int) -> pd.Series:
    return series.shift(lag_periods)

def shorten_common_prefix(x_name: str, y_name: str):
    x_prefix = x_name.split()[0]
    y_prefix = y_name.split()[0]
    if x_prefix == y_prefix:
        return x_prefix, x_name[len(x_prefix)+1:], y_name[len(x_prefix)+1:]
    return '', x_name, y_name
# ---------------------------------------------------------------------
# 6. Core engine – build figs, tables, and diagnostics dict
# ---------------------------------------------------------------------
def build_panel(
    df,
    x_variable, x_frequency, x_transforms, x_lag,
    y_variable, y_frequency, y_transforms, y_lag,
    fit_type, dual_axis_flags, diagnostic_choices,
    native_frequency
):
    """Return:
       scatter_figure, timeseries_figure, summary_panel (Dash HTML)
    """
    base_x_series = apply_transforms_to_series(df[x_variable], x_transforms, native_frequency)
    base_y_series = apply_transforms_to_series(df[y_variable], y_transforms, native_frequency)

    # ---------- Pre‑processing ----------
    common_prefix, _, _ = shorten_common_prefix(x_variable, y_variable)

    transformed_x = apply_lag_to_series(
        apply_transforms_to_series(df[x_variable], x_transforms, native_frequency), x_lag)
    transformed_y = apply_lag_to_series(
        apply_transforms_to_series(df[y_variable], y_transforms, native_frequency), y_lag)

    series_x = transformed_x.resample(frequency_to_rule[x_frequency]).last().dropna()
    series_y = transformed_y.resample(frequency_to_rule[y_frequency]).last().dropna()

    combined_df = pd.concat([series_x.rename(x_variable),
                             series_y.rename(y_variable)], axis=1).dropna()
    # ---------- Regression ----------
    design_matrix   = sm.add_constant(combined_df[x_variable])
    regression_model = sm.OLS(combined_df[y_variable], design_matrix).fit()

    # Fit‑line for scatter
    if fit_type == 'linear':
        slope, intercept = np.polyfit(combined_df[x_variable],
                                      combined_df[y_variable], 1)
        predictions = slope * combined_df[x_variable] + intercept
        eqn_text = f"y = {slope:.3f}x + {intercept:.3f}"
    elif fit_type == 'exp':
        mask = combined_df[y_variable] > 0
        log_slope, log_int = np.polyfit(combined_df.loc[mask, x_variable],
                                        np.log(combined_df.loc[mask, y_variable]), 1)
        a = np.exp(log_int)
        predictions = a * np.exp(log_slope * combined_df[x_variable])
        eqn_text = f"y = {a:.3f}·e^({log_slope:.3f}x)"
    else:  # quadratic
        a2,b2,c2 = np.polyfit(combined_df[x_variable], combined_df[y_variable], 2)
        predictions = a2*combined_df[x_variable]**2 + b2*combined_df[x_variable] + c2
        eqn_text = f"y = {a2:.3f}x² + {b2:.3f}x + {c2:.3f}"

    r_squared = 1 - np.sum((combined_df[y_variable] - predictions)**2) / \
                    np.sum((combined_df[y_variable] - combined_df[y_variable].mean())**2)

    # ---------- Scatter figure ----------
    # Add Date as a column for hover info
    combined_df = combined_df.copy()
    combined_df['Date'] = combined_df.index.strftime('%Y-%m-%d')
    scatter_fig = px.scatter(combined_df, x=x_variable, y=y_variable, custom_data=['Date', x_variable, y_variable])
    x_rng = np.linspace(combined_df[x_variable].min(),
                        combined_df[x_variable].max(), 200)
    if fit_type == 'linear':
        y_fit = slope * x_rng + intercept
    elif fit_type == 'exp':
        y_fit = a * np.exp(log_slope * x_rng)
    else:
        y_fit = a2*x_rng**2 + b2*x_rng + c2
    scatter_fig.add_trace(go.Scatter(x=x_rng, y=y_fit,
                                     mode='lines', line=dict(dash='dash'),
                                     showlegend=False))
    scatter_fig.add_annotation(xref='paper', yref='paper',
                               x=0.05, y=0.95,
                               text=f"{eqn_text}<br>R² = {r_squared:.3f}",
                               showarrow=False)
    scatter_fig.update_traces(
        hovertemplate="Date: %{customdata[0]}<br>" +
                      f"{x_variable}: %{{customdata[1]:.3f}}<br>" +
                      f"{y_variable}: %{{customdata[2]:.3f}}<extra></extra>"
    )
    scatter_fig.update_layout(margin={'t':30})

    # ---------- Time‑series figure ----------
    union_df = pd.concat([transformed_x.rename(x_variable),
                          transformed_y.rename(y_variable)], axis=1).dropna(how='all')
    separate_axes = 'dual' in dual_axis_flags
    if separate_axes:
        ts_fig = go.Figure()
        ts_fig.add_trace(go.Scatter(x=union_df.index, y=union_df[x_variable],
                                    name=x_variable))
        ts_fig.add_trace(go.Scatter(x=union_df.index, y=union_df[y_variable],
                                    name=y_variable, yaxis='y2'))
        ts_fig.update_layout(yaxis2=dict(overlaying='y', side='right'),
                             margin={'t':30})
    else:
        ts_fig = px.line(union_df, x=union_df.index,
                         y=[x_variable, y_variable])
        ts_fig.update_layout(margin={'t':30})

    # ---------- Diagnostics ----------
    slope_coef = float(regression_model.params[x_variable])
    p_val_slope = float(regression_model.pvalues[x_variable])
    dw_stat = float(durbin_watson(regression_model.resid))
    bp_stat, bp_p, _, _ = het_breuschpagan(regression_model.resid, design_matrix)
    jb_stat, jb_p, _, _ = jarque_bera(regression_model.resid)

    # Granger causality
    gc_df = pd.concat([series_y.rename(y_variable),
                       series_x.rename(x_variable)], axis=1).dropna()
    max_lag = min(4, len(gc_df)-1)
    if max_lag >= 1:
        gc_results = grangercausalitytests(gc_df, maxlag=max_lag, verbose=False)
    else:
        gc_results = {}

    # Unit root
    adf_x_s, adf_x_p = adfuller(series_x, autolag='AIC')[:2]
    kpss_x_s, kpss_x_p = kpss(series_x, regression='c', nlags='auto')[:2]
    adf_y_s, adf_y_p = adfuller(series_y, autolag='AIC')[:2]
    kpss_y_s, kpss_y_p = kpss(series_y, regression='c', nlags='auto')[:2]

    unit_root_results = [
        {"variable":"X","test":"ADF",  "stat":adf_x_s, "p_value":adf_x_p,
         "stationary": adf_x_p<0.05},
        {"variable":"X","test":"KPSS", "stat":kpss_x_s, "p_value":kpss_x_p,
         "stationary": kpss_x_p>=0.05},
        {"variable":"Y","test":"ADF",  "stat":adf_y_s, "p_value":adf_y_p,
         "stationary": adf_y_p<0.05},
        {"variable":"Y","test":"KPSS", "stat":kpss_y_s, "p_value":kpss_y_p,
         "stationary": kpss_y_p>=0.05},
    ]

    # Cointegration
    ci_df = pd.concat([series_x.rename(x_variable),
                       series_y.rename(y_variable)], axis=1).dropna()
    res_eg = sm.OLS(ci_df[y_variable],
                    sm.add_constant(ci_df[x_variable])).fit().resid
    eg_stat, eg_p = adfuller(res_eg, autolag='AIC')[:2]
    joh = coint_johansen(ci_df, det_order=0, k_ar_diff=1)
    joh_trace, joh_crit95 = float(joh.lr1[0]), float(joh.cvt[0,1])

    cointegration_results = {
        "engle_granger": {"stat": eg_stat, "p_value": eg_p,
                          "cointegrated": eg_p<0.05},
        "johansen_trace": {"stat": joh_trace, "crit_95": joh_crit95,
                           "cointegrated": joh_trace>joh_crit95}
    }

    # ---------- Diagnostic chart figures ----------
    diag_figs = []
    diag_annots = {}
    for name in diagnostic_choices or []:
        if name == 'R² vs Lag':
            base_df = pd.concat(
                [
                    base_x_series
                        .resample(frequency_to_rule[x_frequency]).last()
                        .rename(x_variable),
                    base_y_series
                        .resample(frequency_to_rule[y_frequency]).last()
                        .rename(y_variable),
                ],
                axis=1
            ).dropna()

            fig, annot = helper.r2_vs_lag(
                base_df,
                x_variable,
                y_variable,
                _period_map['YoY'][x_frequency]
            )
        elif name == 'Residuals vs Fitted':
            fig, annot = helper.residuals_vs_fitted(regression_model)
        elif name == 'Residuals Histogram':
            fig, annot = helper.residuals_histogram(regression_model.resid)
        elif name == 'Normal Q–Q Plot':
            fig, annot = helper.normal_qq_plot(regression_model.resid)
        elif name == 'Residuals ACF':
            fig, annot = helper.residuals_acf(regression_model.resid)
        elif name == 'Residuals PACF':
            fig, annot = helper.residuals_pacf(regression_model.resid)
        elif name == 'Scale–Location Plot':
            fig, annot = helper.scale_location_plot(regression_model)
        elif name == 'Influence Plot':
            fig, annot = helper.influence_plot(regression_model)
        elif name == 'CUSUM of Residuals':
            fig, annot = helper.cusum_plot(regression_model.resid)
        else:
            continue
        diag_figs.append(fig)
        diag_annots[name] = annot

    # ---------- Dash summary panel (tables + charts) ----------
    head = lambda txt: html.Th(txt, style={'border':'1px solid','padding':'6px',
                                           'fontWeight':'bold','backgroundColor':'#f0f0f0'})
    cell = lambda txt: html.Td(txt, style={'border':'1px solid','padding':'6px',
                                           'textAlign':'center'})

    # Durbin-Watson interpretation
    if dw_stat >= 1.8 and dw_stat <= 2.2:
        dw_interp = "No Autocorrelation"
    elif dw_stat < 1.8:
        dw_interp = "Positive Autocorrelation"
    else:
        dw_interp = "Negative Autocorrelation"

    # Jarque-Bera interpretation
    jb_interp = "Normal Residuals" if jb_p >= 0.05 else "Non-Normal Residuals"

    regression_rows = [
        html.Tr([head(c) for c in ['Test','Stat','p‑value','Interpret']]),
        html.Tr([cell('Slope Coefficient'),
                 cell(f"{slope_coef:.4f}"),
                 cell(f"{p_val_slope:.4f}"),
                 cell("Significant" if p_val_slope<0.05 else "Not significant")]),
        html.Tr([cell('Durbin–Watson'), cell(f"{dw_stat:.3f}"), cell(''),
                 cell(dw_interp)]),
        html.Tr([cell('Breusch–Pagan'), cell(f"{bp_stat:.3f}"), cell(f"{bp_p:.3f}"),
                 cell("Heteroskedastic" if bp_p<0.05 else "Homoskedastic")]),
        html.Tr([cell('Jarque–Bera'), cell(f"{jb_stat:.3f}"), cell(f"{jb_p:.3f}"),
                 cell(jb_interp)])
    ]

    gc_head = [head(f'Lag ({frequency_to_unit[x_frequency]})'),
               head('p‑value'), head('Interpret')]
    gc_rows_html = [html.Tr(gc_head)]
    if gc_results:
        for lag in range(1, max_lag+1):
            p_gc = gc_results[lag][0]['ssr_ftest'][1]
            gc_rows_html.append(html.Tr([
                cell(str(lag)),
                cell(f"{p_gc:.3f}"),
                cell("Reject H₀" if p_gc<0.05 else "Fail to reject")
            ]))
    else:
        gc_rows_html.append(html.Tr([cell('-'), cell('n/a'), cell('Insufficient data')]))

    unit_head = [head(c) for c in ['Variable','Test','Stat','p‑value','Interpret']]
    unit_rows_html = [html.Tr(unit_head)]
    for ur in unit_root_results:
        unit_rows_html.append(
            html.Tr([cell('X' if ur['variable']=='X' else 'Y'),
                     cell(ur['test']),
                     cell(f"{ur['stat']:.3f}"),
                     cell(f"{ur['p_value']:.3f}"),
                     cell("Stationary" if ur['stationary'] else "Non‑stationary")])
        )

    ci_head = [head(c) for c in ['Test','Stat','Critical (95%)','Interpret']]
    ci_rows_html = [
        html.Tr(ci_head),
        html.Tr([cell('Engle–Granger'), cell(f"{eg_stat:.3f}"), cell(''),
                 cell("Cointegrated" if eg_p<0.05 else "Not cointegrated")]),
        html.Tr([cell('Johansen Trace'), cell(f"{joh_trace:.3f}"),
                 cell(f"{joh_crit95:.3f}"),
                 cell("Cointegrated" if joh_trace>joh_crit95 else "Not cointegrated")])
    ]

    summary_panel = html.Div([
        html.H3('Observations'),
        html.Div(f"{x_variable}: {len(series_x)} obs"),
        html.Div(f"{y_variable}: {len(series_y)} obs"),
        html.H3('Regression Diagnostics'),
        html.Table(regression_rows, style={'borderCollapse':'collapse','width':'100%'}),
        html.H3(f'Granger Causality ({x_variable} → {y_variable})'),
        html.Table(gc_rows_html, style={'borderCollapse':'collapse','width':'100%','marginTop':'20px'}),
        html.H3('Unit Root Tests'),
        html.Table(unit_rows_html, style={'borderCollapse':'collapse','width':'100%','marginTop':'20px'}),
        html.H3('Cointegration Tests'),
        html.Table(ci_rows_html, style={'borderCollapse':'collapse','width':'100%','marginTop':'20px'}),
        html.H3('Diagnostic Charts'),
        html.Div(
            [dcc.Graph(figure=f, style={'height':'200px'}) for f in diag_figs],
            style={'display':'grid','gridTemplateColumns':'repeat(3,1fr)','gap':'20px'}
        )
    ])

    return scatter_fig, ts_fig, summary_panel
# ---------------------------------------------------------------------
# 7. Dash layout
# ---------------------------------------------------------------------
app = dash.Dash(__name__)

app.layout = html.Div(style={'width':'90%','margin':'auto'}, children=[

    # Title + download button (absolute‑positioned)
    html.Div(style={'position':'relative','paddingBottom':'1rem'}, children=[
        html.H1('Time Series Scatter & Diagnostics', style={'textAlign':'center'}),
        html.Button('Download Panel as JSON', id='download-json-btn', n_clicks=0,
                    style={'position':'absolute','top':0,'right':0,
                           'padding':'0.5rem 1rem'}),
        dcc.Download(id='download-json')
    ]),

    # Two‑column layout (controls/plots | stats)
    html.Div(style={'display':'flex','alignItems':'flex-start',
                    'justifyContent':'space-between'}, children=[

        # LEFT: controls + plots
        html.Div(style={'width':'60%'}, children=[
            # Upload block --------------------------------------------------
            html.Div(style={'marginBottom':'20px'}, children=[
                html.Label('Upload Excel DataFrame', style={'fontWeight':'bold'}),
                dcc.Upload(
                    id='upload-data',
                    children=html.Button('Select File'),
                    multiple=False,
                    accept='.xlsx',
                    style={'marginBottom':'10px'}
                ),
                dcc.Store(id='stored-data'),
                html.Div(id='upload-status', style={'fontSize':'0.9em', 'color':'#888'})
            ]),

            # X & Y control blocks --------------------------------------------------
            html.Div(style={'display':'flex','justifyContent':'space-between',
                            'marginBottom':'20px'}, children=[

                # X controls
                html.Div(style={'width':'48%','display':'flex','flexDirection':'column','gap':'8px'}, children=[
                    html.Label('X Variable'),
                    dcc.Dropdown(id='x-axis-dropdown', options=[], value=None, clearable=False),
                    html.Label('X Frequency'),
                    dcc.Dropdown(id='x-freq-dropdown', options=[], value=None, clearable=False),
                    html.Label('X Transformations'),
                    dcc.Dropdown(id='x-transform-dropdown', options=[], value=[], multi=True),
                    html.Label('X Lag Periods'),
                    dcc.Input(id='x-lag-periods', type='number', value=0, min=0, step=1)
                ]),

                # Y controls
                html.Div(style={'width':'48%','display':'flex','flexDirection':'column','gap':'8px'}, children=[
                    html.Label('Y Variable'),
                    dcc.Dropdown(id='y-axis-dropdown', options=[], value=None, clearable=False),
                    html.Label('Y Frequency'),
                    dcc.Dropdown(id='y-freq-dropdown', options=[], value=None, clearable=False),
                    html.Label('Y Transformations'),
                    dcc.Dropdown(id='y-transform-dropdown', options=[], value=[], multi=True),
                    html.Label('Y Lag Periods'),
                    dcc.Input(id='y-lag-periods', type='number', value=0, min=0, step=1)
                ])
            ]),

            # Scatter‑plot block ----------------------------------------------------
            html.Div(style={'display':'flex','marginBottom':'20px','border':'1px solid #ccc',
                            'padding':'10px','borderRadius':'5px'}, children=[
                html.Div(style={'width':'20%','paddingRight':'10px',
                                'display':'flex','flexDirection':'column','gap':'8px'}, children=[
                    html.Label('Scatter Plot Options', style={'fontWeight':'bold'}),
                    html.Label('Best Fit Line'),
                    dcc.Dropdown(id='scatter-fit-dropdown',
                                 options=[{'label':'Linear','value':'linear'},
                                          {'label':'Exponential','value':'exp'},
                                          {'label':'Quadratic','value':'quad'}],
                                 value='linear', clearable=False)
                ]),
                html.Div(style={'width':'80%'}, children=[dcc.Graph(id='scatter-plot')])
            ]),

            # Time‑series block -----------------------------------------------------
            html.Div(style={'display':'flex','marginBottom':'20px','border':'1px solid #ccc',
                            'padding':'10px','borderRadius':'5px'}, children=[
                html.Div(style={'width':'20%','paddingRight':'10px',
                                'display':'flex','flexDirection':'column','gap':'8px'}, children=[
                    html.Label('Time Series Plot Options', style={'fontWeight':'bold'}),
                    dcc.Checklist(id='dual-axis-checkbox',
                                  options=[{'label':'Separate Y axes','value':'dual'}],
                                  value=[])
                ]),
                html.Div(style={'width':'80%'}, children=[dcc.Graph(id='ts-plot')])
            ])
        ]),

        # RIGHT: statistical analysis panel ---------------------------------------
        html.Div(style={'width':'38%','paddingLeft':'20px'}, children=[
            html.H2('Statistical Analysis', style={'textAlign':'center'}),
            # Remove diagnostic chart dropdown and label
            html.Div(id='regression-summary', style={'marginTop':'20px'})
        ])
    ])
])
# ---------------------------------------------------------------------
# 8. Callback – update plots & panel
# ---------------------------------------------------------------------
@app.callback(
    Output('scatter-plot','figure'),
    Output('ts-plot','figure'),
    Output('regression-summary','children'),
    Input('scatter-fit-dropdown','value'),
    Input('x-axis-dropdown','value'),
    Input('x-freq-dropdown','value'),
    Input('x-transform-dropdown','value'),
    Input('x-lag-periods','value'),
    Input('y-axis-dropdown','value'),
    Input('y-freq-dropdown','value'),
    Input('y-transform-dropdown','value'),
    Input('y-lag-periods','value'),
    Input('dual-axis-checkbox','value'),
    State('stored-data', 'data')
)
def update_all_plots_and_analysis(
    fit_type,
    x_var, x_freq, x_tx, x_lag,
    y_var, y_freq, y_tx, y_lag,
    dual_flags,
    stored_json
):
    # Use uploaded data if present
    if stored_json:
        df = pd.read_json(stored_json, orient='split')
        allowed_variables, allowed_frequencies, transform_options, native_frequency = get_allowed_from_df(df)
    else:
        return go.Figure(), go.Figure(), html.Div("Please upload an Excel file to begin.")
    # Defensive: check for valid variable selection
    if not x_var or not y_var or x_var not in df.columns or y_var not in df.columns:
        return go.Figure(), go.Figure(), html.Div("Please select valid X and Y variables.")
    diag_choices = [n for n in helper.chart_functions.keys() if n != 'Rolling R²']
    scatter_fig, ts_fig, summary_panel = build_panel(
        df,
        x_var, x_freq, x_tx, x_lag,
        y_var, y_freq, y_tx, y_lag,
        fit_type, dual_flags, diag_choices,
        native_frequency
    )
    return scatter_fig, ts_fig, summary_panel

@app.callback(
    Output('stored-data', 'data'),
    Output('upload-status', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def parse_upload(contents, filename):
    if contents is None:
        return dash.no_update, ''
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        df = pd.read_excel(io.BytesIO(decoded), parse_dates=['Date'])
        df.set_index('Date', inplace=True)
        msg = f"Loaded: {filename} ({df.shape[0]} rows, {df.shape[1]} columns)"
        return df.to_json(date_format='iso', orient='split'), msg
    except Exception as e:
        return dash.no_update, f"Error loading file: {e}"

# Helper to get allowed variables and options from a DataFrame
def get_allowed_from_df(df):
    allowed_prefixes = ('USA','EUR','JPN','SWI','DEU','FRA','ESP','ITA','CAN','AUS','WLD')
    allowed_variables = [c for c in df.columns if c.startswith(allowed_prefixes)]
    date_diffs = df.index.to_series().diff().dt.days.dropna()
    median_days = date_diffs.median() if not date_diffs.empty else 1
    if median_days <= 1:
        native_frequency = 'daily'
    elif median_days <= 7:
        native_frequency = 'weekly'
    elif median_days <= 31:
        native_frequency = 'monthly'
    elif median_days <= 92:
        native_frequency = 'quarterly'
    else:
        native_frequency = 'annual'
    allowed_frequencies = allowed_frequencies_by_native[native_frequency]
    transform_options = [inflation_adjust_option] + [
        opt for opt in other_transform_options
        if opt['value'] in allowed_transformations_by_frequency[native_frequency]
    ]
    return allowed_variables, allowed_frequencies, transform_options, native_frequency

@app.callback(
    Output('x-axis-dropdown', 'options'),
    Output('y-axis-dropdown', 'options'),
    Output('x-freq-dropdown', 'options'),
    Output('y-freq-dropdown', 'options'),
    Output('x-transform-dropdown', 'options'),
    Output('y-transform-dropdown', 'options'),
    Output('x-axis-dropdown', 'value'),
    Output('y-axis-dropdown', 'value'),
    Output('x-freq-dropdown', 'value'),
    Output('y-freq-dropdown', 'value'),
    Output('x-transform-dropdown', 'value'),
    Output('y-transform-dropdown', 'value'),
    Input('stored-data', 'data'),
)
def update_dropdowns(stored_json):
    if stored_json:
        df = pd.read_json(stored_json, orient='split')
        allowed_variables, allowed_frequencies, transform_options, native_frequency = get_allowed_from_df(df)
        # Defaults
        x_var = allowed_variables[0] if allowed_variables else None
        y_var = allowed_variables[1] if len(allowed_variables) > 1 else (allowed_variables[0] if allowed_variables else None)
        freq = allowed_frequencies[0] if allowed_frequencies else None
        tx = ['YoY % Change'] if any('YoY % Change' in o['value'] for o in transform_options) else []
        return (
            [{'label':v,'value':v} for v in allowed_variables],
            [{'label':v,'value':v} for v in allowed_variables],
            [{'label':f.capitalize(),'value':f} for f in allowed_frequencies],
            [{'label':f.capitalize(),'value':f} for f in allowed_frequencies],
            transform_options,
            transform_options,
            x_var,
            y_var,
            freq,
            freq,
            tx,
            tx
        )
    # If no data, return empty dropdowns
    return [], [], [], [], [], [], None, None, None, None, [], []
# ---------------------------------------------------------------------
# 9. Callback – JSON download
# ---------------------------------------------------------------------
# The JSON download logic is now handled by stats_webapp_json.py
# ---------------------------------------------------------------------
# 10. Run
# ---------------------------------------------------------------------
if __name__ == '__main__':
    # No initial data_frame or native_frequency, so pass None or dummy values if needed
    register_json_download_callback(app, None, frequency_to_rule, _period_map, None)
    app.run(debug=True)
