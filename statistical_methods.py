import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM
from statsmodels.tsa.api import VAR
from arch.unitroot import PhillipsPerron

# ----------------------------------------
# 1. Unit‐root / stationarity tests
# ----------------------------------------
def adf_test(series, name='', alpha=0.10):
    """Augmented Dickey–Fuller test (null = unit root)."""
    stat, p_value, _, _, crit_values, _ = adfuller(series.dropna(), autolag='AIC')
    stars = ''.join('*' for thresh in [0.01, 0.05, 0.10] if p_value < thresh)
    decision = f"reject{stars}" if p_value < alpha else "accept"
    return stat, p_value, decision

def kpss_test(series, name='', alpha=0.10):
    """KPSS test (null = stationarity)."""
    stat, p_value, _, crit_values = kpss(series.dropna(), nlags='auto')
    stars = ''.join('*' for thresh in [0.01, 0.05, 0.10] if p_value < thresh)
    decision = f"reject{stars}" if p_value < alpha else "accept"
    return stat, p_value, decision

def pp_test(series, name='', alpha=0.10):
    """Phillips–Perron test (null = unit root)."""
    pp = PhillipsPerron(series.dropna())
    stat, p_value = pp.stat, pp.pvalue
    stars = ''.join('*' for thresh in [0.01, 0.05, 0.10] if p_value < thresh)
    decision = f"reject{stars}" if p_value < alpha else "accept"
    return stat, p_value, decision

def stationarity_report(series, name='Series'):
    """
    Runs ADF and KPSS on the series and prints a combined report.
    """
    print(f"\n=== Stationarity Report for {name} ===")
    stat, p, dec = adf_test(series, name=name)
    print(f"ADF: stat={stat:.4f}, p={p:.4f} → {dec}")
    stat, p, dec = kpss_test(series, name=name)
    print(f"KPSS: stat={stat:.4f}, p={p:.4f} → {dec}")

# ----------------------------------------
# 2. Granger‐causality test (5‐step manual)
# ----------------------------------------
def granger_causality(x, y, maxlag=4, alpha=0.05):
    """
    Manual Granger‐causality test x → y.
    Example:
        granger_causality(df['Bank Loan Growth'], df['GDP Growth'], maxlag=4)
    """
    print("\n=== Granger‐Causality Test ===")
    print("1) Stationarity of originals:")
    stationarity_report(x, 'X')
    stationarity_report(y, 'Y')

    print("\n2) Difference non-stationary series:")
    x_s = x.diff().dropna()
    y_s = y.diff().dropna()
    stationarity_report(x_s, 'X (diff)')
    stationarity_report(y_s, 'Y (diff)')

    data = pd.concat([y_s, x_s], axis=1).dropna()
    y_aligned, x_aligned = data.iloc[:, 0], data.iloc[:, 1]

    for lag in range(1, maxlag+1):
        print(f"\n--- Lag = {lag} ---")
        # restricted: Y ~ lags of Y
        Ylags = pd.concat([y_aligned.shift(i) for i in range(1, lag+1)], axis=1).dropna()
        Ylags.columns = [f"Y_lag{i}" for i in range(1, lag+1)]
        y_r = y_aligned.loc[Ylags.index]
        mod_r = sm.OLS(y_r, sm.add_constant(Ylags)).fit()
        RSSR = (mod_r.resid**2).sum()

        # unrestricted: Y ~ lags of Y + lags of X
        Xlags = pd.concat([x_aligned.shift(i) for i in range(1, lag+1)], axis=1)
        Xlags.columns = [f"X_lag{i}" for i in range(1, lag+1)]
        df_ur = pd.concat([Ylags, Xlags], axis=1).dropna()
        y_ur = y_aligned.loc[df_ur.index]
        mod_ur = sm.OLS(y_ur, sm.add_constant(df_ur)).fit()
        RSSUR = (mod_ur.resid**2).sum()

        m, n = lag, len(y_ur)
        k = int(mod_ur.df_model) + 1
        F = ((RSSR - RSSUR)/m) / (RSSUR/(n - k))
        from scipy.stats import f
        p_val = 1 - f.cdf(F, m, n - k)
        print(f"F={F:.4f}, p={p_val:.4f} → ", end='')
        print("reject" if p_val < alpha else "accept")

# ----------------------------------------
# 3. Johansen cointegration test
# ----------------------------------------
def johansen_cointegration_test(df, det_order=0, k_ar_diff=1):
    """
    Johansen’s cointegration test.
    Example:
        johansen_cointegration_test(df[['A','B','C']], det_order=0, k_ar_diff=1)
    """
    print("\n=== Johansen Cointegration Test ===")
    jres = coint_johansen(df.dropna(), det_order, k_ar_diff)
    for i, trace in enumerate(jres.lr1):
        crits = jres.cvt[i]
        print(f"r<={i}: trace={trace:.4f}, cv(90/95/99)={crits}")
    return jres

# ----------------------------------------
# 4. Six‐step VAR analysis
# ----------------------------------------
def var_six_step(df, maxlags=8, ic='hqic'):
    """
    1) Transform
    2) Unit roots
    3) Breaks
    4) Lag order (by {ic})
    5) Fit VAR
    6) IRFs
    """
    print("\n=== VAR Six-Step Procedure ===")
    for col in df.columns:
        stationarity_report(df[col], col)

    model = VAR(df.dropna())
    sel = model.select_order(maxlags)
    lag = getattr(sel, ic)
    print(f"Chosen lag by {ic}: {lag}")

    res = model.fit(lag)
    print(res.summary())

    print("\nImpulse Responses:")
    irf = res.irf(10)
    irf.plot(orth=False)
    plt.show()
    return res

# ----------------------------------------
# 5. Six‐step VECM analysis
# ----------------------------------------
def vecm_six_step(df, det_order=0, k_ar_diff=1):
    """
    1) Transform
    2) Unit roots
    3) Breaks
    4) Determine k_ar_diff
    5) Johansen → rank r
    6) Fit VECM
    """
    print("\n=== VECM Six-Step Procedure ===")
    for col in df.columns:
        stationarity_report(df[col], col)

    jres = coint_johansen(df.dropna(), det_order, k_ar_diff)
    r = sum(jres.lr1 > jres.cvt[:,1])
    print(f"Rank at 95%: r = {r}")

    vec = VECM(df.dropna(), k_ar_diff=k_ar_diff, coint_rank=r, deterministic='ci')
    res = vec.fit()
    print("\nAlpha:")
    print(res.alpha)
    print("\nBeta:")
    print(res.beta)
    return res

# ----------------------------------------
# 6. Original regression‐report generator
# ----------------------------------------
def generate_regression_report(excel_path, pdf_path):
    data = pd.read_excel(excel_path, parse_dates=['Date'])
    data.set_index('Date', inplace=True)
    df = data[['GDP Growth', 'Bank Loan Growth', 'Public Debt Growth']].dropna()

    X = sm.add_constant(df[['Bank Loan Growth', 'Public Debt Growth']])
    y = df['GDP Growth']
    model = sm.OLS(y, X).fit()

    resid, fitted = model.resid, model.fittedvalues
    vif_df = pd.DataFrame({
        'variable': X.columns,
        'VIF': [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    })
    shapiro_stat, shapiro_p = stats.shapiro(resid)
    jb_stat, jb_p, _, _ = sm.stats.jarque_bera(resid)
    _, lm_p, _, f_p = het_breuschpagan(resid, X)
    infl = model.get_influence()
    cooks, _ = infl.cooks_distance

    lr = LinearRegression(fit_intercept=False)
    kf = KFold(n_splits=5, shuffle=False)
    mse = -cross_val_score(lr, df[['Bank Loan Growth','Public Debt Growth']], y,
                           scoring='neg_mean_squared_error', cv=kf)

    # bootstrap
    n_boot = 1000
    boot = np.zeros((n_boot, X.shape[1]))
    rng = np.random.default_rng(42)
    for i in range(n_boot):
        samp = df.sample(frac=1, replace=True, random_state=rng)
        Xb = sm.add_constant(samp[['Bank Loan Growth','Public Debt Growth']])
        yb = samp['GDP Growth']
        mb = sm.OLS(yb, Xb).fit()
        boot[i,:] = mb.params.values
    ci_low = np.percentile(boot, 2.5, axis=0)
    ci_high= np.percentile(boot,97.5, axis=0)

    with PdfPages(pdf_path) as pdf:
        # Title page
        fig = plt.figure(figsize=(8,11))
        fig.text(0.5,0.6,'Multiple Linear Regression Report',ha='center',fontsize=20,weight='bold')
        fig.text(0.5,0.5,'Predicting GDP Growth using Bank Loan & Public Debt Growth',
                 ha='center',fontsize=12)
        pdf.savefig(fig); plt.close(fig)

        # Summary
        fig = plt.figure(figsize=(8,11)); plt.axis('off')
        fig.text(0.01,0.99,model.summary().as_text(),va='top',fontsize=8,family='monospace')
        pdf.savefig(fig); plt.close(fig)

        # VIF
        fig = plt.figure(figsize=(8,11)); plt.axis('off')
        fig.text(0.01,0.99,'Variance Inflation Factors\n\n'+vif_df.to_string(index=False),
                 va='top',fontsize=10)
        pdf.savefig(fig); plt.close(fig)

        # Residuals vs Fitted
        fig, ax = plt.subplots(figsize=(8,4))
        ax.scatter(fitted,resid,edgecolor='k'); ax.axhline(0,linestyle='--'); 
        ax.set(title='Residuals vs. Fitted', xlabel='Fitted',ylabel='Residuals')
        pdf.savefig(fig); plt.close(fig)

        # QQ Plot
        fig = plt.figure(figsize=(8,4))
        sm.qqplot(resid,line='45',fit=True,ax=plt.gca()); plt.title('Normal Q–Q')
        pdf.savefig(fig); plt.close(fig)

        # Normality & Heterosk
        fig = plt.figure(figsize=(8,4)); plt.axis('off')
        txt = (f'Shapiro: stat={shapiro_stat:.4f}, p={shapiro_p:.4f}\n'
               f'Jarque–Bera: stat={jb_stat:.4f}, p={jb_p:.4f}\n\n'
               f'BP LM p={lm_p:.4f}, F p={f_p:.4f}')
        fig.text(0.01,0.99,'Normality & Heteroskedasticity\n\n'+txt,va='top',fontsize=10)
        pdf.savefig(fig); plt.close(fig)

        # Cook's D
        fig, ax = plt.subplots(figsize=(8,4))
        ax.stem(df.index, cooks,markerfmt='.',linefmt='C0-')
        ax.set(title="Cook's Distance",xlabel='Date',ylabel="Cook's D")
        pdf.savefig(fig); plt.close(fig)

        # CV
        fig = plt.figure(figsize=(8,4)); plt.axis('off')
        cvtxt = f'5-fold CV MSE: {np.round(mse,4).tolist()}\nMean: {mse.mean():.4f}'
        fig.text(0.01,0.99,'Cross-Validation\n\n'+cvtxt,va='top',fontsize=10)
        pdf.savefig(fig); plt.close(fig)

        # Bootstrap CIs
        fig = plt.figure(figsize=(8,4)); plt.axis('off')
        boottxt = 'Bootstrap 95% CIs\n\n'
        for idx,name in enumerate(model.params.index):
            boottxt += f'{name}: [{ci_low[idx]:.4f}, {ci_high[idx]:.4f}]\n'
        fig.text(0.01,0.99,boottxt,va='top',fontsize=10)
        pdf.savefig(fig); plt.close(fig)

    print(f"Report saved as {pdf_path}")

# ----------------------------------------
# 7. Unit‐root summary table (screenshot)
# ----------------------------------------
def unit_root_summary_table(df, var, periods):
    """
    Prints ADF, KPSS, PP over level, 1st and 2nd diffs,
    for each (label, start, end) in periods.
    """
    s = df[var]
    transforms = [("Level", s), ("1st Diff", s.diff()), ("2nd Diff", s.diff().diff())]
    labels = [lbl for lbl,_,_ in periods]
    # Header
    print(f"\nUnit-root tests for {var}\n")
    hdr = f"{'Series':<8}  {'Transform':<10}  {'Test':<5}  " + "  ".join(f"{lab:<12}" for lab in labels)
    print(hdr)
    # Rows
    for trans_name, series in transforms:
        for test_name, func in [("ADF", adf_test), ("KPSS", kpss_test), ("PP", pp_test)]:
            parts = [
                var if test_name=="ADF" else "",
                trans_name if test_name=="ADF" else "",
                test_name
            ]
            for _, start, end in periods:
                stat, p, dec = func(series.loc[start:end], name=var)
                parts.append(dec)
            row = f"{parts[0]:<8}  {parts[1]:<10}  {parts[2]:<5}  " + \
                  "  ".join(f"{p:<12}" for p in parts[3:])
            print(row)


# ----------------------------------------
# Example usage for screenshot only
# ----------------------------------------
if __name__ == "__main__":
    # Load data
    data = pd.read_excel("data_quarterly.xlsx", parse_dates=["Date"])
    data.set_index("Date", inplace=True)

    # Compute ratios for screenshot
    data["NFAP/PDY"]      = data["NFAP"] / data["PDY"]
    data["log(NFAP/PDY)"] = np.log(data["NFAP/PDY"])

    # Define sub-samples exactly as in the screenshot
    periods = [
        ("1960–2003", "1960-01-01", "2003-12-31"),
        ("2004–2016", "2004-01-01", "2016-12-31"),
    ]

    # Reproduce only the screenshot’s unit-root tables
    for var in ["NFAP", "PDY", "NFAP/PDY", "log(NFAP/PDY)"]:
        unit_root_summary_table(data, var, periods)


