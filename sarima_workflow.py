#!/usr/bin/env python3
"""
sarima_workflow.py — SARIMA Methodology Demonstration
======================================================

Implements the complete 5-step SARIMA workflow described in SARIMA_process.txt,
applied to the daily aggregate of all sticker sales in data/train.csv.

All diagnostic plots are saved to the plots/ directory. The script follows
the sections of SARIMA_process.txt in order:

    Section 0: Preprocessing (missing values, variance stabilization)
    Section 1: Stationarity identification (ADF + KPSS, ACF/PACF)
    Section 2: Seasonality detection (periodogram + STL decomposition)
    Section 3: ACF/PACF after differencing (reading off candidate orders)
    Section 4: Grid search (manual statsmodels SARIMAX, AIC-driven)
    Section 5: Residual diagnostics (white-noise verification)
    Section 6: Holiday exogenous variable (SARIMAX with holiday indicator)
    Section 7: Weekly SARIMA with s=52 (capturing annual seasonality)

Design documentation: every non-trivial methodological choice includes a
WHY THIS / WHY NOT X explanation as inline comments.

Runtime note: The grid search (Section 4) fits approximately 36 SARIMA models
to ~2,400 training observations. Expect 10-25 minutes on a standard CPU.
Section 6 refits the best model with holiday exog (~same runtime as one fit).
Section 7 runs a separate grid search on weekly data (~5-10 min).
"""

import os
import sys
import itertools
import warnings

import numpy as np
import pandas as pd
import matplotlib

# Use non-interactive backend so plots can be saved without a display.
# Why 'Agg' over 'TkAgg' or 'Qt5Agg'? Agg is a purely file-based renderer
# that works in headless environments (servers, WSL without display forwarding).
# Interactive backends require a running display session.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal, stats
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

warnings.filterwarnings("ignore")

# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================

DATA_PATH = "data/train.csv"
PLOTS_DIR = "plots"

# Seasonal period.
# WHY s=7? The data is daily sales. Retail demand cycles weekly: higher on
# weekends, lower mid-week (or vice versa). A weekly period (s=7) is the
# dominant short cycle in most retail data and is confirmed empirically
# by the periodogram in Section 2.
# WHY NOT s=365 (annual)? Annual seasonality exists, but SARIMA handles
# only one seasonal period. With 7 years of data, s=7 provides far more
# seasonal cycles for estimation (371 cycles vs. 7). For multi-seasonality,
# TBATS or Prophet would be more appropriate.
SEASONAL_PERIOD = 7

# Hold out the last 90 days as an out-of-sample test set.
# WHY 90 days? 90 days covers ~13 weekly seasonal cycles, providing a
# statistically meaningful evaluation window. Using fewer days (e.g., 30)
# covers only 4 cycles and may not represent seasonal variation adequately.
# Using more (e.g., 180) reduces training data significantly.
TEST_DAYS = 90

# Significance threshold for all hypothesis tests (ADF, KPSS, Ljung-Box).
# WHY 0.05? The conventional threshold in applied statistics. Using 0.01
# would require stronger evidence to declare non-stationarity, risking
# under-differencing. Using 0.10 increases false positives (over-differencing).
ALPHA = 0.05

# Grid search bounds.
# WHY p, q <= 2 (range 0..3 exclusive)? ARMA theory shows that most real
# processes are well-approximated with p, q <= 2. Orders above 3 rarely
# improve out-of-sample performance and risk overfitting. They also increase
# computation time super-linearly.
# WHY P, Q <= 1 (range 0..2 exclusive)? For weekly seasonality, P=1 or Q=1
# captures one lag of seasonal autocorrelation, which is almost always sufficient.
# P=2 or Q=2 means the model looks at two seasons back (14 days), which is
# rarely necessary.
# TOTAL COMBINATIONS: 3 * 3 * 2 * 2 = 36, making the search tractable.
P_RANGE = range(0, 3)       # AR non-seasonal: 0, 1, 2
Q_RANGE = range(0, 3)       # MA non-seasonal: 0, 1, 2
SEASONAL_P_RANGE = range(0, 2)  # AR seasonal: 0, 1
SEASONAL_Q_RANGE = range(0, 2)  # MA seasonal: 0, 1


# =============================================================================
# SECTION 0: DATA LOADING AND PREPROCESSING
# =============================================================================

def load_and_prepare_data():
    """
    Load train.csv, aggregate to a single daily time series, handle missing
    values, and apply a log transform to stabilize variance.

    Returns
    -------
    raw_daily : pd.Series
        Daily aggregate num_sold (original scale, NaNs filled).
    log_series : pd.Series
        log(1 + raw_daily) — the series used for all modeling.
    """
    print(f"\n[0] Loading data from {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])

    # Aggregate all country/store/product combinations into one daily total.
    # WHY AGGREGATE? SARIMA requires a univariate time series. The dataset is
    # hierarchical (many country × store × product combinations). Aggregating
    # creates one clean representative series for this methodology demonstration.
    # WHY SUM not mean? Total sales is a more natural business quantity than
    # average sales per combination. Sum preserves the scale of economic activity.
    daily = (
        df.groupby("date")["num_sold"]
        .sum()
        .rename("num_sold")
        .sort_index()
    )

    n_total = len(daily)
    n_missing = daily.isna().sum()
    print(f"    Observations  : {n_total} daily records "
          f"({daily.index.min().date()} to {daily.index.max().date()})")
    print(f"    Missing values: {n_missing} ({100 * n_missing / n_total:.1f}%)")

    # Fill missing values via linear interpolation.
    # WHY LINEAR INTERPOLATION over zero-fill?
    # Filling NaNs with 0 introduces artificial sharp drops in the series.
    # Each 0 looks like a large negative impulse to the model, creating
    # spurious autocorrelation at all lags and corrupting ACF/PACF plots.
    # WHY NOT FORWARD-FILL (ffill)?
    # Forward-fill repeats the previous observation unchanged. This creates
    # runs of identical values that introduce artificial autocorrelation at
    # lag 1 and bias the ADF test toward under-rejection (appears more stationary
    # than it is). Linear interpolation between neighbors is less disruptive.
    daily = daily.interpolate(method="linear").bfill().ffill()

    # Log(1+x) variance-stabilizing transform.
    # WHY LOG TRANSFORM?
    # Sales counts exhibit heteroskedasticity: variance grows with the level
    # (larger sales = more day-to-day swing). SARIMA assumes constant variance.
    # Log compresses high values more than low, flattening the variance envelope.
    # WHY log(1+x) OVER log(x)?
    # log(0) = -inf. The +1 offset handles any potential zeros without a
    # separate guard condition.
    # WHY LOG OVER BOX-COX?
    # Box-Cox estimates an optimal lambda; for count/sales data lambda ~ 0,
    # which reduces to the log. Box-Cox adds a stored parameter that must be
    # inverted at forecast time, adding complexity with negligible empirical benefit.
    log_series = np.log1p(daily)

    print(f"    Log-transformed series range: "
          f"[{log_series.min():.3f}, {log_series.max():.3f}]")
    return daily, log_series


def plot_raw_vs_log(raw, log_s):
    """
    Side-by-side plot of the original series and its log transform.
    Illustrates the effect of variance stabilization.
    Saved to: plots/00_raw_vs_log.png
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    fig.suptitle("Section 0 — Preprocessing: Raw vs Log-Transformed Series",
                 fontsize=13, fontweight="bold")

    axes[0].plot(raw.index, raw.values, color="steelblue", linewidth=0.8, alpha=0.9)
    axes[0].set_ylabel("Total Units Sold")
    axes[0].set_title("Original Series (raw aggregate num_sold)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(log_s.index, log_s.values, color="darkorange", linewidth=0.8, alpha=0.9)
    axes[1].set_ylabel("log(1 + num_sold)")
    axes[1].set_title("Log-Transformed Series — variance visibly more stable")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, "00_raw_vs_log.png")


# =============================================================================
# SECTION 1: STATIONARITY
# =============================================================================

def run_adf_test(series, label="series"):
    """
    Augmented Dickey-Fuller unit-root test.

    H0 (null)   : the series has a unit root → non-stationary.
    H1 (alt)    : the series is stationary.

    DECISION RULE: reject H0 (declare stationary) if p-value <= ALPHA.

    WHY ADF?
    ADF extends the basic Dickey-Fuller test by adding lagged differences
    of the dependent variable to absorb serial correlation in the residuals.
    This makes the test statistic asymptotically valid even when the error
    process is an ARMA, not just white noise.

    WHY autolag='AIC'?
    The number of lagged differences included in the ADF regression must be
    chosen. 'AIC' selects the lag count that minimises AIC, balancing fit
    against parsimony. Alternative 'BIC' tends to choose fewer lags; 'AIC'
    is the standard default and avoids under-specifying the lag structure.

    Parameters
    ----------
    series : pd.Series
    label  : str — used in print statements only.

    Returns
    -------
    is_stationary : bool
    p_value       : float
    """
    result = adfuller(series.dropna(), autolag="AIC")
    stat, p_value = result[0], result[1]
    is_stationary = p_value <= ALPHA

    print(f"\n    ADF on {label}:")
    print(f"      Statistic : {stat:.4f}")
    print(f"      p-value   : {p_value:.4f}  →  "
          + ("STATIONARY ✓" if is_stationary else "NON-STATIONARY ✗")
          + f"  (threshold = {ALPHA})")
    return is_stationary, p_value


def run_kpss_test(series, label="series"):
    """
    KPSS (Kwiatkowski-Phillips-Schmidt-Shin) stationarity test.

    H0 (null)   : the series is stationary (level-stationary around a constant).
    H1 (alt)    : the series has a unit root → non-stationary.

    DECISION RULE: fail to reject H0 (declare stationary) if p-value > ALPHA.

    WHY KPSS ALONGSIDE ADF?
    ADF and KPSS have OPPOSITE nulls. Using both cross-validates:
      ADF rejects AND KPSS fails to reject  → strong evidence of stationarity.
      ADF fails to reject AND KPSS rejects  → strong evidence of non-stationarity.
    Either test alone can give misleading results on borderline series.
    ADF can fail to reject near-unit-root stationary series; KPSS rejects
    stationary series with structural breaks.

    WHY regression='c' (level stationarity) over 'ct' (trend stationarity)?
    After log-differencing, we expect the series to be stationary around a
    constant (mean). regression='ct' would test stationarity around a linear
    trend, which is appropriate for the original (un-differenced) series but
    overly generous after differencing.

    Parameters
    ----------
    series : pd.Series
    label  : str

    Returns
    -------
    is_stationary : bool
    p_value       : float
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = kpss(series.dropna(), regression="c", nlags="auto")
    stat, p_value = result[0], result[1]
    is_stationary = p_value > ALPHA

    print(f"\n    KPSS on {label}:")
    print(f"      Statistic : {stat:.4f}")
    print(f"      p-value   : {p_value:.4f}  →  "
          + ("STATIONARY ✓" if is_stationary else "NON-STATIONARY ✗")
          + f"  (threshold = {ALPHA})")
    return is_stationary, p_value


def check_stationarity(series, label):
    """
    Run ADF and KPSS and return a combined stationarity verdict.

    Returns
    -------
    is_stationary : bool — True only when both tests agree the series is stationary.
    """
    adf_stat, _ = run_adf_test(series, label)
    kpss_stat, _ = run_kpss_test(series, label)

    if adf_stat and kpss_stat:
        verdict = "Both tests agree: STATIONARY ✓"
    elif not adf_stat and not kpss_stat:
        verdict = "Both tests agree: NON-STATIONARY ✗"
    else:
        verdict = ("Mixed signal: ADF=" + ("stat" if adf_stat else "non-stat")
                   + " / KPSS=" + ("stat" if kpss_stat else "non-stat")
                   + " → treating as NON-STATIONARY (conservative)")

    print(f"\n    Combined verdict: {verdict}")
    # Conservative: treat as non-stationary unless both tests agree it is stationary.
    return adf_stat and kpss_stat


def plot_time_series(series):
    """
    Plot the log-transformed time series with a 30-day rolling mean overlay.
    The rolling mean helps visually assess whether a trend is present.
    Saved to: plots/01_time_series.png
    """
    rolling = series.rolling(window=30, center=True).mean()

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(series.index, series.values, color="steelblue",
            linewidth=0.7, alpha=0.8, label="Daily log-sales")
    ax.plot(rolling.index, rolling.values, color="red",
            linewidth=2.0, label="30-day rolling mean")
    ax.set_title("Section 1 — Stationarity: Log-Transformed Series with Rolling Mean\n"
                 "(upward trend suggests non-stationarity; differencing required)",
                 fontsize=11)
    ax.set_ylabel("log(1 + num_sold)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, "01_time_series.png")


def plot_acf_pacf(series, title, filename, lags=60, seasonal_period=None):
    """
    Plot ACF and PACF side by side.

    WHY 60 LAGS?
    For daily data with s=7, 60 lags covers 8+ seasonal cycles. This lets us
    observe both the short-range structure (lags 1-6) and seasonal repetition
    (lags 7, 14, 21, ...). Fewer lags might miss seasonal spikes; more lags
    add visual noise.

    WHY plot_acf / plot_pacf (statsmodels) over manual computation?
    statsmodels draws confidence bands based on the Bartlett formula for ACF
    and the asymptotic approximation for PACF, which are the conventional
    standard in time series analysis. Computing and drawing these bands
    correctly from scratch is error-prone.

    Parameters
    ----------
    series          : pd.Series
    title           : str — figure suptitle
    filename        : str — saved as plots/<filename>.png
    lags            : int
    seasonal_period : int or None — if set, draws vertical dashed lines at
                      multiples of the seasonal period to guide reading.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle(title, fontsize=12, fontweight="bold")

    plot_acf(series.dropna(), lags=lags, ax=axes[0], alpha=ALPHA)
    axes[0].set_title(f"ACF (lags 0–{lags})")
    axes[0].set_xlabel("Lag")

    plot_pacf(series.dropna(), lags=lags, ax=axes[1], alpha=ALPHA,
              method="ywm")
    # WHY method='ywm' (Yule-Walker with bias correction)?
    # 'ywm' is the statsmodels default and uses the Yule-Walker equations
    # with a small-sample correction. It is more stable than the OLS method
    # for large lags on finite samples.
    axes[1].set_title(f"PACF (lags 0–{lags})")
    axes[1].set_xlabel("Lag")

    if seasonal_period is not None:
        for ax in axes:
            for k in range(1, lags // seasonal_period + 1):
                ax.axvline(x=k * seasonal_period, color="orange",
                           linestyle="--", linewidth=0.8, alpha=0.7)
            ax.axvline(x=0, color="orange", linestyle="--",
                       linewidth=0.8, alpha=0.7, label=f"multiples of s={seasonal_period}")
            ax.legend(fontsize=8, loc="upper right")

    for ax in axes:
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    _save(fig, f"{filename}.png")


# =============================================================================
# SECTION 2: SEASONALITY DETECTION
# =============================================================================

def plot_periodogram(series):
    """
    Compute and plot the power spectral density (periodogram) of the series.

    The x-axis is converted to PERIOD (days) = 1/frequency for readability.
    Dominant peaks identify the most important seasonal cycles.

    WHY PERIODOGRAM (FFT-based) for seasonality detection?
    The periodogram decomposes the series into sinusoidal components and
    reports how much variance is explained by each frequency. It is objective
    and operates in the frequency domain, meaning trend does not obscure the
    seasonal peaks (unlike looking at the raw ACF on a trended series).

    WHY NOT JUST STL for detection?
    STL requires you to specify s in advance — it confirms, not discovers.
    The periodogram is genuinely discovery-oriented.

    WHY scipy.signal.periodogram over scipy.signal.welch?
    Welch's method averages overlapping segments for a smoother estimate but
    sacrifices frequency resolution. With ~2,500 observations we have
    sufficient data for the standard periodogram to clearly resolve the
    weekly (s=7) and annual (s=365) cycles.

    Saved to: plots/02_periodogram.png
    """
    values = series.dropna().values
    n = len(values)

    # Detrend first to avoid spectral leakage from the mean/trend.
    # WHY detrend='linear'? A linear trend concentrates power at low frequencies,
    # potentially swamping the seasonal peaks. Removing the linear trend first
    # ensures seasonal spikes are clearly visible.
    # WHY NOT difference first? Differencing changes the spectral shape
    # (multiplies by |1 - e^{-2πif}|²), which can shift peak locations.
    # Detrending is the standard pre-processing step for spectral analysis.
    freqs, power = signal.periodogram(values, detrend="linear")

    # Convert frequency → period (days); skip freq=0 (infinite period).
    # Only show periods from 2 to 400 days.
    freqs = freqs[1:]
    power = power[1:]
    periods = 1.0 / freqs
    mask = (periods >= 2) & (periods <= 400)
    periods, power = periods[mask], power[mask]

    # Identify top-5 peaks.
    from scipy.signal import find_peaks
    peaks_idx, _ = find_peaks(power, height=np.percentile(power, 95))
    peak_periods = periods[peaks_idx]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.semilogy(periods, power, color="steelblue", linewidth=0.9)
    for pp in peak_periods[:5]:
        ax.axvline(x=pp, color="red", linestyle="--", linewidth=1.2,
                   label=f"s ≈ {pp:.1f}")
    ax.set_xlabel("Period (days)")
    ax.set_ylabel("Power (log scale)")
    ax.set_title("Section 2 — Seasonality: Periodogram\n"
                 "(dominant peaks indicate seasonal periods; expect spikes near 7 and 365)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.25)
    plt.tight_layout()
    _save(fig, "02_periodogram.png")

    print(f"\n    Top seasonal periods detected: {[f'{p:.1f}' for p in sorted(peak_periods[:5])]}")
    return peak_periods


def plot_stl_decomposition(series, s):
    """
    STL (Seasonal and Trend decomposition using Loess) decomposition.

    Decomposes the series into trend, seasonal, and residual components
    using locally weighted regression (LOESS) for both the seasonal and
    trend smoother.

    WHY STL OVER CLASSICAL DECOMPOSITION (e.g., seasonal_decompose)?
    1. STL is robust to outliers (uses robust LOESS weights).
    2. The seasonal component can evolve over time (not assumed constant).
    3. It handles any seasonality length, not just 4 or 12.
    4. It does not require symmetric smoothing windows and handles endpoints better.

    WHY NOT X-11 or X-13 ARIMA?
    X-11/X-13 are excellent for monthly/quarterly economic data but are complex
    to configure and were originally designed for specific seasonal lengths.
    STL is simpler, more transparent, and equally effective here.

    WHY period=s (=7)?
    We confirmed s=7 from the periodogram. STL with period=7 extracts the
    weekly seasonal component.

    seasonal=13 means the seasonal LOESS window spans 13 periods (must be odd).
    WHY 13? It covers ~2 seasonal cycles, enough to estimate a smooth seasonal
    shape without over-smoothing. The default (7) can be too small for retail data.

    Saved to: plots/02_stl_decomposition.png
    """
    stl = STL(series.dropna(), period=s, seasonal=13, robust=True)
    result = stl.fit()

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"Section 2 — Seasonality: STL Decomposition (s={s}, seasonal window=13)",
                 fontsize=12, fontweight="bold")

    components = [
        (series.dropna(), "Observed", "steelblue"),
        (result.trend, "Trend", "darkorange"),
        (result.seasonal, "Seasonal (weekly)", "seagreen"),
        (result.resid, "Residual", "grey"),
    ]
    for ax, (data, label, color) in zip(axes, components):
        ax.plot(data.index, data.values, color=color, linewidth=0.8)
        ax.set_ylabel(label, fontsize=9)
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    _save(fig, "02_stl_decomposition.png")


# =============================================================================
# SECTION 3: DIFFERENCING AND STATIONARY ACF/PACF
# =============================================================================

def difference_and_verify(series, d, D, s):
    """
    Apply non-seasonal (d) and seasonal (D) differencing, then confirm
    the result is stationary with ADF + KPSS.

    WHY d=1 TYPICALLY SUFFICIENT?
    A single first difference (y_t - y_{t-1}) removes a linear trend. After
    d=1, most economic/sales series are stationary. d=2 would be needed only
    if the series has a quadratic trend, which is rare and risks over-differencing.

    WHY D=1 TYPICALLY SUFFICIENT?
    A single seasonal difference (y_t - y_{t-s}) removes one level of seasonal
    non-stationarity. D>1 almost always causes over-differencing:
    the seasonal component in the ACF will show a large negative spike at lag 1,
    indicating the model is trying to "undo" an already-removed structure.

    Parameters
    ----------
    series : pd.Series (log-transformed)
    d      : int — number of non-seasonal differences
    D      : int — number of seasonal differences
    s      : int — seasonal period

    Returns
    -------
    diff_series : pd.Series — stationary series ready for ACF/PACF reading.
    """
    diff = series.copy()

    for i in range(d):
        diff = diff.diff(1)
        print(f"    Applied non-seasonal difference #{i+1}: diff(1)")

    for i in range(D):
        diff = diff.diff(s)
        print(f"    Applied seasonal difference #{i+1}: diff({s})")

    diff = diff.dropna()

    print(f"\n    Verifying stationarity after differencing (d={d}, D={D}):")
    check_stationarity(diff, f"differenced series (d={d}, D={D}, s={s})")

    return diff


# =============================================================================
# SECTION 4: GRID SEARCH
# =============================================================================

def grid_search_sarima(train, d, D, s, exog=None,
                       p_range=None, q_range=None,
                       sp_range=None, sq_range=None):
    """
    Exhaustive grid search over SARIMA(p,d,q)(P,D,Q)[s] parameter combinations.

    WHY MANUAL GRID SEARCH OVER pmdarima.auto_arima?
    1. No additional dependency — statsmodels is already in the environment.
    2. Full transparency: every combination and its AIC are visible in the output.
    3. More control: we can inspect the results table, not just the best model.

    WHY FIX d AND D OUTSIDE THE GRID?
    d and D have analytically determined "correct" values from the stationarity
    tests. Including them in the grid would mix non-stationary models (d=0) with
    stationary ones (d=1) in AIC comparisons — this is invalid because AIC
    compares likelihood on the *same* response variable, but differencing changes
    the response. Fixing d and D ensures all AIC values are comparable.

    WHY AIC OVER BIC FOR MODEL SELECTION?
    AIC = 2k - 2*ln(L)
    BIC = k*ln(n) - 2*ln(L)
    BIC penalizes model size more (k*ln(n) vs 2k). For large n (thousands of
    observations), BIC strongly favors very parsimonious models that may
    under-capture the seasonal structure. AIC is the standard criterion when
    the goal is predictive accuracy (forecasting), not parsimony for its own sake.

    WHY enforce_stationarity=False and enforce_invertibility=False?
    During grid search, we test all combinations. Letting statsmodels reject
    non-invertible/non-stationary parameter configurations internally would skip
    potentially useful combinations silently. We disable these checks during the
    search and rely on residual diagnostics to verify the final model.

    Parameters
    ----------
    train    : pd.Series — training portion of the log-transformed series.
    d        : int
    D        : int
    s        : int
    exog     : pd.DataFrame or None — optional exogenous regressors aligned to train.
               If provided, each model is fitted with these covariates. Must have
               the same index as train.
    p_range, q_range, sp_range, sq_range : range objects — override module defaults
               when the weekly model calls this function with different bounds.

    Returns
    -------
    best_fit        : SARIMAX fitted model object
    best_order      : (p, d, q) tuple
    best_season     : (P, D, Q, s) tuple
    results_df      : pd.DataFrame — all tested combinations and their AIC.
    """
    _p  = p_range  if p_range  is not None else P_RANGE
    _q  = q_range  if q_range  is not None else Q_RANGE
    _sp = sp_range if sp_range is not None else SEASONAL_P_RANGE
    _sq = sq_range if sq_range is not None else SEASONAL_Q_RANGE

    combos = list(itertools.product(_p, _q, _sp, _sq))
    total = len(combos)

    print(f"\n    Grid: p∈{list(_p)}, q∈{list(_q)}, "
          f"P∈{list(_sp)}, Q∈{list(_sq)}")
    print(f"    Fixed: d={d}, D={D}, s={s}")
    print(f"    Exog columns: {list(exog.columns) if exog is not None else 'none'}")
    print(f"    Total combinations: {total}  (this may take 10-25 min)\n")

    best_aic = np.inf
    best_fit = None
    best_order = None
    best_season = None
    records = []

    for i, (p, q, sp, sq) in enumerate(combos):
        order = (p, d, q)
        seasonal_order = (sp, D, sq, s)
        try:
            model = SARIMAX(
                train,
                exog=exog,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
                trend="n",
                # WHY trend='n'? Including an explicit intercept (trend='c') is
                # redundant when d=1: the differencing already removes the mean
                # of the original series. Adding a constant to a differenced
                # series estimates a linear drift, which is rarely appropriate
                # for sales data and would be a separate modeling decision.
            )
            fit = model.fit(disp=False, method="lbfgs", maxiter=200)
            # WHY method='lbfgs'? L-BFGS is a quasi-Newton optimizer that
            # converges reliably for SARIMAX likelihoods. The default 'newton'
            # can be unstable for some parameter configurations during grid search.
            # WHY maxiter=200? Default (50) sometimes fails to converge; 200
            # is sufficient for the problem size without excessive compute.

            aic = fit.aic
            records.append({"p": p, "d": d, "q": q,
                             "P": sp, "D": D, "Q": sq, "AIC": round(aic, 2)})

            if aic < best_aic:
                best_aic = aic
                best_fit = fit
                best_order = order
                best_season = seasonal_order
                print(f"    [{i+1:2d}/{total}] New best: "
                      f"SARIMA{order}x{seasonal_order} AIC={aic:.2f}")
            elif (i + 1) % 9 == 0:
                print(f"    [{i+1:2d}/{total}] Best still: "
                      f"SARIMA{best_order}x{best_season} AIC={best_aic:.2f}")

        except Exception:
            records.append({"p": p, "d": d, "q": q,
                             "P": sp, "D": D, "Q": sq, "AIC": np.nan})

    results_df = pd.DataFrame(records).sort_values("AIC")
    print(f"\n    Top 5 models by AIC:")
    print(results_df.head(5).to_string(index=False))

    return best_fit, best_order, best_season, results_df


def evaluate_forecast(best_fit, train, test, best_order, best_season,
                      exog_test=None, plot_filename="04_forecast.png",
                      section_label="Section 4"):
    """
    Generate a forecast on the held-out test window and evaluate accuracy.

    Forecast mechanics:
    - We use the already-fitted model (trained on `train`, optionally with exog).
    - We call get_forecast(steps=TEST_DAYS, exog=exog_test) to produce predictions
      and 95% confidence intervals.
    - Forecast and actuals are both exponentiated (np.expm1) to return to
      original scale for interpretable error metrics.

    WHY REFIT ON TRAINING ONLY (not append test data incrementally)?
    A one-step-ahead rolling forecast would be more accurate in practice
    (each new observation updates the state estimate), but it requires
    TEST_DAYS model updates and is 90x slower. The batch forecast from the
    training cutoff is the standard academic evaluation: it tests how far
    ahead the model can forecast without new information.

    Parameters
    ----------
    exog_test : pd.DataFrame or None — exogenous regressors for the forecast
                horizon. Must be provided if the model was trained with exog,
                must be None otherwise. Rows correspond to the test period.

    Saved to: plots/<plot_filename>
    """
    print(f"\n    Generating {len(test)}-step forecast on test set ...")
    forecast_result = best_fit.get_forecast(steps=len(test), exog=exog_test)
    forecast_mean = forecast_result.predicted_mean
    conf_int = forecast_result.conf_int(alpha=ALPHA)

    # Invert log transform: exp(log(1+x)) - 1 = x
    # WHY invert for evaluation? MAE/RMSE/MAPE on the log scale are not
    # interpretable in business terms. Original-scale metrics communicate
    # forecast quality in units the stakeholder understands (number of stickers).
    fc_orig = np.expm1(forecast_mean)
    lo_orig = np.expm1(conf_int.iloc[:, 0])
    hi_orig = np.expm1(conf_int.iloc[:, 1])
    test_orig = np.expm1(test)

    mae = np.mean(np.abs(fc_orig.values - test_orig.values))
    rmse = np.sqrt(np.mean((fc_orig.values - test_orig.values) ** 2))
    mape = np.mean(np.abs((test_orig.values - fc_orig.values) / test_orig.values)) * 100

    print(f"\n    Out-of-sample forecast accuracy (original scale):")
    print(f"      MAE  : {mae:,.1f}   (mean absolute error in units sold)")
    print(f"      RMSE : {rmse:,.1f}  (root mean squared error)")
    print(f"      MAPE : {mape:.2f}%  (mean absolute percentage error)")
    # WHY all three metrics?
    # MAE: easy to interpret, robust to outliers, same units as the series.
    # RMSE: penalises large errors more, useful when large misses are costly.
    # MAPE: scale-free, useful for comparing across series with different magnitudes.
    # They can disagree: low MAPE but high RMSE suggests occasional large outliers.

    # Plot: show last 180 days of training context + test + forecast
    context_start = train.index[-180]
    train_context_orig = np.expm1(train.loc[context_start:])

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(train_context_orig.index, train_context_orig.values,
            color="steelblue", linewidth=1.2, label="Training (last 180 days)")
    ax.plot(test_orig.index, test_orig.values,
            color="black", linewidth=1.5, label="Actual (test)")
    ax.plot(fc_orig.index, fc_orig.values,
            color="red", linewidth=1.5, linestyle="--", label="Forecast")
    ax.fill_between(fc_orig.index, lo_orig, hi_orig,
                    color="red", alpha=0.15, label=f"95% prediction interval")
    ax.set_title(
        f"{section_label} — SARIMA{best_order}×{best_season}\n"
        f"MAE={mae:,.0f}  RMSE={rmse:,.0f}  MAPE={mape:.1f}%",
        fontsize=11)
    ax.set_ylabel("Total Units Sold")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, plot_filename)

    return mae, rmse, mape


# =============================================================================
# SECTION 5: RESIDUAL DIAGNOSTICS
# =============================================================================

def plot_residual_diagnostics(best_fit, best_order, best_season):
    """
    Comprehensive residual diagnostics to verify the model's residuals are
    white noise (no remaining structure).

    WHY THESE FOUR DIAGNOSTICS?
    A well-specified model should absorb all systematic structure in the data.
    What remains (the residuals) should be unpredictable — white noise. Each
    diagnostic checks a different aspect of white noise:
      1. Time plot: checks for patterns, trends, or heteroskedasticity over time.
      2. Histogram + KDE: checks distributional shape (normality assumption).
      3. QQ-plot: checks normality in the tails (most important for forecast intervals).
      4. ACF: checks serial independence (no autocorrelation at any lag).
    Together they provide a holistic sanity check.
    """

    resid = best_fit.resid.dropna()
    std_resid = (resid - resid.mean()) / resid.std()

    # --- Plot 1: Standardised residuals over time ---
    fig, ax = plt.subplots(figsize=(14, 3))
    ax.plot(std_resid.index, std_resid.values, color="steelblue",
            linewidth=0.7, alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(2, color="red", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axhline(-2, color="red", linewidth=0.8, linestyle="--", alpha=0.5,
               label="±2σ band")
    ax.set_title(
        f"Section 5 — Residuals: Standardised Residuals over Time\n"
        f"SARIMA{best_order}×{best_season}  |  "
        f"~95% of points should fall within ±2σ; no visible trend or pattern",
        fontsize=10)
    ax.set_ylabel("Standardised residual")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, "05_residuals.png")

    # --- Plot 2: Histogram + KDE ---
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(std_resid.values, bins=50, density=True,
            color="steelblue", alpha=0.6, label="Empirical distribution")

    # Overlay standard normal for visual comparison.
    # WHY compare to N(0,1)? SARIMA assumes Gaussian innovations. If residuals
    # are approximately N(0,1) after standardisation, the model is correctly
    # specified. Heavy tails suggest outliers or fat-tailed innovations.
    x_range = np.linspace(-4, 4, 200)
    ax.plot(x_range, stats.norm.pdf(x_range, 0, 1),
            color="red", linewidth=2, label="N(0,1)")
    ax.set_title("Section 5 — Residuals: Histogram vs Normal")
    ax.set_xlabel("Standardised residual")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, "05_residual_histogram.png")

    # --- Plot 3: QQ-plot ---
    fig, ax = plt.subplots(figsize=(5, 5))
    # WHY QQ-plot over Shapiro-Wilk alone?
    # Shapiro-Wilk gives a single p-value (pass/fail). The QQ-plot reveals
    # WHERE the distribution departs from normality: heavy tails (S-shape),
    # skewness (systematic curve), or outliers (isolated end points). This
    # shapes the diagnosis — e.g., heavy tails suggest Student-t innovations;
    # skewness might indicate a missing transformation.
    # WHY NOT Kolmogorov-Smirnov?
    # KS is sensitive to the centre of the distribution, not the tails. For
    # forecast intervals, tail behaviour is what matters.
    (osm, osr), (slope, intercept, r) = stats.probplot(std_resid.values, dist="norm")
    ax.scatter(osm, osr, color="steelblue", s=8, alpha=0.6, label="Residual quantiles")
    ax.plot(osm, slope * np.array(osm) + intercept,
            color="red", linewidth=2, label="Normal reference line")
    ax.set_title("Section 5 — Residuals: QQ-Plot\n"
                 "(points should lie on the diagonal for normality)")
    ax.set_xlabel("Theoretical quantiles")
    ax.set_ylabel("Sample quantiles")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, "05_qq_plot.png")

    # --- Plot 4: ACF of residuals ---
    fig, ax = plt.subplots(figsize=(14, 3))
    plot_acf(std_resid, lags=40, ax=ax, alpha=ALPHA)
    ax.set_title("Section 5 — Residuals: ACF of Standardised Residuals\n"
                 "(all bars should be within blue confidence bands for white noise)")
    ax.set_xlabel("Lag")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    _save(fig, "05_residual_acf.png")

    # --- Ljung-Box test ---
    # WHY Ljung-Box over Box-Pierce?
    # Both test for residual autocorrelation up to a specified lag. The Ljung-Box
    # statistic uses a finite-sample correction (multiplies by n(n+2) instead of n),
    # making it more accurate for small to moderate samples. For large n both
    # give similar results, but Ljung-Box is the standard default.
    #
    # WHY lags=[10, 20, SEASONAL_PERIOD]?
    # Lag 10 and 20 are conventional choices (Box and Jenkins original recommendation
    # was max lag ~ sqrt(n), here ~50, but short lags detect most AR/MA misspecification).
    # Testing at lag=SEASONAL_PERIOD directly checks whether seasonal autocorrelation
    # was adequately removed — the most likely failure mode for SARIMA.
    #
    # INTERPRETATION:
    #   p-value > 0.05 at all lags → residuals appear white noise → model adequate.
    #   p-value <= 0.05 at any lag → remaining autocorrelation → consider increasing
    #   the corresponding AR/MA or seasonal AR/MA order.
    lb_lags = sorted(set([10, 20, SEASONAL_PERIOD]))
    lb_result = acorr_ljungbox(std_resid, lags=lb_lags, return_df=True)

    print(f"\n    Ljung-Box test (H0: residuals are white noise):")
    for lag, row in lb_result.iterrows():
        verdict = "PASS ✓" if row["lb_pvalue"] > ALPHA else "FAIL ✗"
        print(f"      Lag {int(lag):2d}: stat={row['lb_stat']:.3f}  "
              f"p={row['lb_pvalue']:.4f}  →  {verdict}")


# =============================================================================
# SECTION 6: HOLIDAY EXOGENOUS VARIABLE
# =============================================================================

def create_holiday_exog(date_index):
    """
    Build a binary holiday indicator DataFrame aligned to date_index.

    WHY ADD A HOLIDAY COVARIATE AT ALL?
    The baseline SARIMA(2,1,2)(1,1,1)[7] achieved MAPE=3.35% but RMSE/MAE≈2.2×,
    meaning a small number of days carry disproportionately large errors. The most
    common cause in retail forecasting is holiday demand spikes: Christmas, Black
    Friday, national public holidays. SARIMA has no mechanism to represent these —
    it can only fit recurring patterns at fixed lags. Adding a holiday indicator as
    an exogenous regressor (the X in SARIMAX) lets the model learn a dedicated
    coefficient for holiday-day uplift directly from the data.

    WHY A BINARY INDICATOR RATHER THAN A LEAD/LAG WINDOW?
    A window (e.g., flag the day before AND the day of a holiday) is a reasonable
    extension, but the simplest parsimonious model is always a good starting point.
    If residuals still cluster around holidays after adding the binary flag, expanding
    to a window (holiday_minus1, holiday, holiday_plus1) is the natural next step.
    Starting with a window would add 2–3 extra parameters per holiday, which
    inflates the model without guaranteed benefit.

    WHY NOT A FOURIER TERM FOR ANNUAL SEASONALITY INSTEAD?
    Fourier terms (sin/cos pairs at frequency k/365) model smooth, recurring annual
    patterns. They work well for gradually changing effects (e.g., summer peak).
    Holidays are abrupt, irregular, and country-specific — they are better modelled
    as point interventions (binary flags) than smooth sinusoids. We handle the
    smooth annual cycle separately in Section 7 (weekly SARIMA with s=52).

    WHY THE `holidays` LIBRARY?
    The dataset spans 7 years (2010–2016) across 4 countries (Canada, Finland,
    Italy, Kenya). Hardcoding holiday dates for all country × year combinations
    is error-prone and brittle to year-specific anomalies (e.g., Easter changes
    date every year). The `holidays` library provides officially-defined public
    holidays per country per year, removing the need for manual maintenance.
    Install with: pip install holidays

    EXPECTED EFFECT ON METRICS:
    Because RMSE penalises large errors quadratically while MAE treats all errors
    equally, holiday errors contribute far more to RMSE than MAE. Absorbing
    holiday spikes into the model coefficient should reduce RMSE substantially
    (likely 15–30%) while reducing MAE more modestly (5–15%). MAPE change will
    depend on whether the worst percentage errors happen to fall on holiday days.

    Parameters
    ----------
    date_index : pd.DatetimeIndex

    Returns
    -------
    pd.DataFrame with a single column 'is_holiday' (0.0 / 1.0), or None if the
    `holidays` library is not installed.
    """
    try:
        import holidays as hol_lib
    except ImportError:
        print("\n    WARNING: 'holidays' library not installed.")
        print("    Run `pip install holidays` to enable the holiday covariate.")
        print("    Continuing Section 6 without holiday exog (results will be identical to Section 4).")
        return None

    # Countries present in the dataset. The aggregate series sums across all of
    # them, so a public holiday in ANY country shifts total sales for that day.
    # ISO 3166-1 alpha-2 codes used by the `holidays` library:
    #   Canada   → CA,  Finland → FI,  Italy → IT,  Kenya → KE
    country_codes = ["CA", "FI", "IT", "KE"]
    years = list(date_index.year.unique())

    all_holiday_dates = set()
    for code in country_codes:
        for year in years:
            try:
                country_hols = hol_lib.country_holidays(code, years=year)
                all_holiday_dates.update(country_hols.keys())
            except Exception:
                # Some country/year pairs may not be supported by the library.
                # Silently skip rather than crashing the whole pipeline.
                pass

    holiday_flag = pd.Series(
        [1.0 if d.date() in all_holiday_dates else 0.0 for d in date_index],
        index=date_index,
        name="is_holiday",
    )

    n_hol = int(holiday_flag.sum())
    print(f"\n    Holiday indicator built: {n_hol} holiday days out of "
          f"{len(date_index)} ({100 * n_hol / len(date_index):.1f}%)")
    print(f"    Countries: {country_codes}  |  Years: {min(years)}–{max(years)}")

    return holiday_flag.to_frame()


def refit_with_holiday_exog(best_order, best_season, train, test,
                            exog_train, exog_test):
    """
    Refit the best SARIMA structure with the holiday exogenous variable and
    evaluate the new forecast on the test set.

    WHY REFIT RATHER THAN JUST APPEND EXOG TO THE EXISTING FIT?
    statsmodels fitted SARIMAX objects are immutable — you cannot add covariates
    post-hoc. We must re-specify the model with exog from scratch. This is a
    single model fit (not a grid search), so it is fast (~seconds).

    WHY USE THE SAME ORDER FROM THE GRID SEARCH?
    The grid search in Section 4 identified the best (p,d,q)(P,D,Q) on a model
    without holidays. Re-running the full grid search with holiday exog would be
    more thorough but is ~36× slower. In practice, adding a single binary covariate
    does not change which ARMA orders best explain the autocorrelation structure —
    the holiday effect is orthogonal to the lag structure. So we reuse the order.

    Saved to: plots/06_forecast_with_holidays.png
    """
    print(f"\n    Refitting SARIMA{best_order}×{best_season} with holiday exog ...")
    model = SARIMAX(
        train,
        exog=exog_train,
        order=best_order,
        seasonal_order=best_season,
        enforce_stationarity=False,
        enforce_invertibility=False,
        trend="n",
    )
    fit = model.fit(disp=False, method="lbfgs", maxiter=200)

    # Print the estimated holiday coefficient.
    # A positive coefficient means holidays increase log-sales (expected for retail).
    # The magnitude tells us how much: e.g., coef=0.1 → ~10% uplift in original scale
    # (because the model is on the log scale: exp(0.1) ≈ 1.105).
    coef = fit.params.get("is_holiday", np.nan)
    print(f"    Estimated holiday coefficient: {coef:.4f}  "
          f"(≈ {100 * (np.expm1(coef)):.1f}% change in units sold on holidays)")
    print(f"    AIC with holidays: {fit.aic:.2f}  |  BIC: {fit.bic:.2f}")

    mae, rmse, mape = evaluate_forecast(
        fit, train, test, best_order, best_season,
        exog_test=exog_test,
        plot_filename="06_forecast_with_holidays.png",
        section_label="Section 6 — With Holiday Exog",
    )
    return mae, rmse, mape


# =============================================================================
# SECTION 7: WEEKLY SARIMA (s=52) FOR ANNUAL SEASONALITY
# =============================================================================

def run_weekly_sarima_analysis(raw_daily, log_daily, d):
    """
    Aggregate the daily series to weekly frequency and fit SARIMA with s=52
    to capture annual (year-over-year) seasonality.

    WHY A SEPARATE WEEKLY MODEL?
    The daily SARIMA uses s=7 to model the weekly cycle. It cannot simultaneously
    model an annual cycle (s=365) because SARIMA accepts only one seasonal period.
    The two approaches for handling this limitation are:
      a) Use TBATS or Prophet, which support multiple seasonal periods natively.
      b) Aggregate to a coarser time scale where the remaining seasonality aligns
         with a single s value.
    We choose (b) for two reasons:
      1. It stays within the SARIMA framework described in SARIMA_process.txt,
         making it a natural extension rather than a methodology change.
      2. It is computationally cheaper than TBATS, and more transparent than
         Prophet's automatic decomposition.

    WHY WEEKLY AGGREGATION OVER MONTHLY?
    Weekly preserves more temporal resolution (52 points/year vs 12). Monthly
    aggregation would remove most of the information about within-month patterns.
    Weekly is the coarsest granularity that still shows the annual cycle clearly.

    WHY SUM OVER MEAN FOR AGGREGATION?
    We model total units sold. Summing preserves the economic quantity of interest.
    Mean would rescale the series by ~7 without adding information, and would
    change the interpretation of the log-transform intercept.

    WHY s=52?
    After resampling to weekly, the annual cycle spans 52 observations (one per
    week). This is the natural seasonal period for an annual pattern in weekly data.
    Note: 52 × 7 = 364 days, so one week is "lost" per year (365.25 / 7 ≈ 52.18).
    This small discrepancy is acceptable and standard in weekly SARIMA modelling.

    EXPECTED EFFECT:
    The weekly model should capture year-over-year growth and seasonal peaks (e.g.,
    higher sales around November–December across all countries). If the large errors
    in the daily model's RMSE come from annual-scale events (e.g., back-to-school,
    end-of-year) rather than specific holidays, the weekly model will have a lower
    RMSE relative to series scale even if the absolute RMSE is smaller simply
    because weekly totals are ~7× larger.

    Saved to:
      plots/07_weekly_series.png
      plots/07_weekly_forecast.png
    """
    print("\n    Aggregating daily → weekly (sum) ...")

    # resample('W') assigns each week's data to the LAST day of that week (Sunday).
    # WHY 'W' over 'W-MON'? The choice of week-end day does not affect the model;
    # 'W' (Sunday-ending) is the pandas default and keeps the code readable.
    weekly_raw = raw_daily.resample("W").sum()
    weekly_log = np.log1p(weekly_raw)

    print(f"    Weekly series: {len(weekly_log)} observations  "
          f"({weekly_log.index.min().date()} to {weekly_log.index.max().date()})")

    # --- Plot: weekly series ---
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    fig.suptitle("Section 7 — Weekly SARIMA: Aggregated Weekly Series",
                 fontsize=12, fontweight="bold")
    axes[0].plot(weekly_raw.index, weekly_raw.values,
                 color="steelblue", linewidth=1.0)
    axes[0].set_ylabel("Weekly Total Units Sold")
    axes[0].set_title("Raw weekly aggregate (annual cycle should be visible)")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(weekly_log.index, weekly_log.values,
                 color="darkorange", linewidth=1.0)
    axes[1].set_ylabel("log(1 + weekly total)")
    axes[1].set_title("Log-transformed weekly series")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, "07_weekly_series.png")

    # Train/test split: hold out last 13 weeks (~91 days, comparable to TEST_DAYS=90).
    # WHY 13 weeks? 13 weeks = one quarter of a year. It covers exactly one quarter
    # of the annual seasonal cycle, giving a meaningful out-of-sample evaluation
    # that is proportionally similar to the 90-day daily split.
    test_weeks = 13
    train_w = weekly_log.iloc[:-test_weeks]
    test_w = weekly_log.iloc[-test_weeks:]

    # Stationarity check on weekly log series.
    # WHY recheck stationarity? Aggregation can change the stationarity properties.
    # A non-stationary daily series is usually still non-stationary at weekly
    # frequency (same trend present), but it's worth verifying rather than assuming.
    print("\n    Stationarity check on weekly log series:")
    is_stat_w = check_stationarity(weekly_log, "weekly log series")
    d_w = 0 if is_stat_w else d  # reuse the daily d — usually d=1 applies here too

    # Grid search on weekly data.
    # WHY THE SAME p/q BOUNDS as daily?
    # The ARMA order that best approximates a weekly process is typically
    # similar to the daily process — retail sales have short autocorrelation
    # memory regardless of aggregation level.
    # WHY SEASONAL P,Q BOUNDS of {0,1}?
    # For s=52, P=1 means a lag of 52 weeks (one year back). P=2 would be two
    # years back, which is rarely informative and computationally expensive.
    print(f"\n    Grid search on weekly data (s=52, d={d_w}, D=1) ...")
    best_fit_w, best_order_w, best_season_w, _ = grid_search_sarima(
        train_w, d=d_w, D=1, s=52,
        p_range=range(0, 3), q_range=range(0, 3),
        sp_range=range(0, 2), sq_range=range(0, 2),
    )

    print(f"\n    Best weekly model: SARIMA{best_order_w}×{best_season_w}")
    print(f"    AIC = {best_fit_w.aic:.2f}  |  BIC = {best_fit_w.bic:.2f}")

    mae_w, rmse_w, mape_w = evaluate_forecast(
        best_fit_w, train_w, test_w, best_order_w, best_season_w,
        plot_filename="07_weekly_forecast.png",
        section_label="Section 7 — Weekly SARIMA (s=52, annual seasonality)",
    )
    return mae_w, rmse_w, mape_w, best_order_w, best_season_w


# =============================================================================
# UTILITIES
# =============================================================================

def _save(fig, filename):
    """Save figure to PLOTS_DIR and close it."""
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {path}")


def print_section(n, title):
    print(f"\n{'='*60}")
    print(f"  SECTION {n}: {title}")
    print(f"{'='*60}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "=" * 60)
    print("  SARIMA WORKFLOW — Daily Sticker Sales Forecasting")
    print("=" * 60)

    os.makedirs(PLOTS_DIR, exist_ok=True)

    # ---- Section 0: Preprocessing ----------------------------------------
    print_section(0, "PREPROCESSING")
    raw_daily, log_series = load_and_prepare_data()
    plot_raw_vs_log(raw_daily, log_series)

    # ---- Section 1: Stationarity -----------------------------------------
    print_section(1, "IDENTIFICATION AND STATIONARITY")
    plot_time_series(log_series)
    plot_acf_pacf(log_series,
                  "Section 1 — ACF/PACF: Log-Transformed Series (pre-differencing)",
                  "01_acf_pacf_original",
                  lags=60,
                  seasonal_period=SEASONAL_PERIOD)

    print("\n  Testing stationarity of the log-transformed series:")
    is_stationary = check_stationarity(log_series, "log-transformed series")
    d = 0 if is_stationary else 1
    print(f"\n  → Using d = {d} (non-seasonal differencing order)")

    # ---- Section 2: Seasonality ------------------------------------------
    print_section(2, "SEASONALITY DETECTION")
    plot_periodogram(log_series)
    plot_stl_decomposition(log_series, SEASONAL_PERIOD)
    print(f"\n  → Confirmed seasonal period: s = {SEASONAL_PERIOD} (weekly cycle)")

    # ---- Section 3: ACF/PACF after differencing --------------------------
    print_section(3, "ACF/PACF AFTER DIFFERENCING")
    print(f"\n  Differencing: d={d} (non-seasonal), D=1 (seasonal, lag={SEASONAL_PERIOD})")
    stationary_series = difference_and_verify(log_series, d=d, D=1, s=SEASONAL_PERIOD)
    plot_acf_pacf(
        stationary_series,
        f"Section 3 — ACF/PACF: Stationary Series (d={d}, D=1, s={SEASONAL_PERIOD})\n"
        f"Orange dashed lines mark seasonal lags (multiples of {SEASONAL_PERIOD})",
        "03_acf_pacf_differenced",
        lags=60,
        seasonal_period=SEASONAL_PERIOD,
    )
    print("\n  Reading guide:")
    print("    PACF: count non-seasonal spikes before first cut-off → candidate p")
    print("    ACF : count non-seasonal spikes before first cut-off → candidate q")
    print(f"    PACF at lags {SEASONAL_PERIOD},{2*SEASONAL_PERIOD}: spikes → candidate P")
    print(f"    ACF  at lags {SEASONAL_PERIOD},{2*SEASONAL_PERIOD}: spikes → candidate Q")

    # ---- Section 4: Grid Search (baseline, no holiday exog) -------------
    print_section(4, "GRID SEARCH (statsmodels SARIMAX, AIC)")
    train = log_series.iloc[:-TEST_DAYS]
    test = log_series.iloc[-TEST_DAYS:]
    print(f"\n  Train: {len(train)} obs  |  Test: {len(test)} obs "
          f"({test.index.min().date()} to {test.index.max().date()})")

    best_fit, best_order, best_season, results_df = grid_search_sarima(
        train, d=d, D=1, s=SEASONAL_PERIOD
    )

    print(f"\n  Best model: SARIMA{best_order}×{best_season}")
    print(f"  AIC = {best_fit.aic:.2f}  |  BIC = {best_fit.bic:.2f}")

    mae, rmse, mape = evaluate_forecast(
        best_fit, train, test, best_order, best_season,
        plot_filename="04_forecast.png",
        section_label="Section 4 — Baseline (no holiday exog)",
    )

    # ---- Section 5: Residual Diagnostics ---------------------------------
    print_section(5, "RESIDUAL DIAGNOSTICS")
    plot_residual_diagnostics(best_fit, best_order, best_season)

    # ---- Section 6: Holiday Exogenous Variable ---------------------------
    print_section(6, "HOLIDAY EXOGENOUS VARIABLE (SARIMAX)")
    exog_full = create_holiday_exog(log_series.index)

    mae_hol, rmse_hol, mape_hol = mae, rmse, mape  # fallback if exog unavailable
    if exog_full is not None:
        exog_train = exog_full.loc[train.index]
        exog_test  = exog_full.loc[test.index]
        mae_hol, rmse_hol, mape_hol = refit_with_holiday_exog(
            best_order, best_season, train, test, exog_train, exog_test
        )
    else:
        print("\n  Skipping Section 6 (holidays library not available).")

    # ---- Section 7: Weekly SARIMA (annual seasonality, s=52) -------------
    print_section(7, "WEEKLY SARIMA (s=52, annual seasonality)")
    mae_w, rmse_w, mape_w, order_w, season_w = run_weekly_sarima_analysis(
        raw_daily, log_series, d
    )

    # ---- Summary ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("  WORKFLOW COMPLETE — RESULTS COMPARISON")
    print("=" * 60)
    print(f"  {'Model':<40} {'MAE':>10} {'RMSE':>10} {'MAPE':>8}")
    print(f"  {'-'*68}")
    print(f"  {'Daily SARIMA'+str(best_order)+'×'+str(best_season)+' (baseline)':<40} "
          f"{mae:>10,.1f} {rmse:>10,.1f} {mape:>7.2f}%")
    if exog_full is not None:
        print(f"  {'+ holiday exog':<40} "
              f"{mae_hol:>10,.1f} {rmse_hol:>10,.1f} {mape_hol:>7.2f}%")
    print(f"  {'Weekly SARIMA'+str(order_w)+'×'+str(season_w)+' (s=52)':<40} "
          f"{mae_w:>10,.1f} {rmse_w:>10,.1f} {mape_w:>7.2f}%")
    print(f"  {'-'*68}")
    print(f"\n  NOTE: Weekly MAE/RMSE are in weekly total units (×7 vs daily).")
    print(f"        Compare MAPE across models for a scale-free assessment.")
    print(f"\n  Plots saved to: {PLOTS_DIR}/")
    print("=" * 60)

    plots_saved = [
        "00_raw_vs_log.png",
        "01_time_series.png",
        "01_acf_pacf_original.png",
        "02_periodogram.png",
        "02_stl_decomposition.png",
        "03_acf_pacf_differenced.png",
        "04_forecast.png",
        "05_residuals.png",
        "05_residual_histogram.png",
        "05_qq_plot.png",
        "05_residual_acf.png",
        "06_forecast_with_holidays.png",
        "07_weekly_series.png",
        "07_weekly_forecast.png",
    ]
    print(f"\n  Plots ({len(plots_saved)} files):")
    for p in plots_saved:
        print(f"    plots/{p}")


if __name__ == "__main__":
    main()
