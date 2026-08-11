import sys
from pathlib import Path
import copy

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
import statsmodels.formula.api as smf
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Lasso
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

PROGRAMS_DIR = Path(__file__).resolve().parents[1]
if str(PROGRAMS_DIR) not in sys.path:
    sys.path.insert(0, str(PROGRAMS_DIR))

from config import ANALYSIS_DATA_DIR, RESULTS_DIR


def significance_stars(pvalue):
    return "***" if pvalue < 0.01 else "**" if pvalue < 0.05 else "*" if pvalue < 0.1 else ""


def run_leave_one_out(model, feature_matrix, outcome_vector):
    loocv = LeaveOneOut()
    predictions = np.zeros(len(outcome_vector))

    for train_idx, test_idx in loocv.split(feature_matrix):
        fitted = copy.deepcopy(model)
        fitted.fit(feature_matrix[train_idx], outcome_vector[train_idx])
        predictions[test_idx] = fitted.predict(feature_matrix[test_idx])

    return predictions


def coef_cell(model, variable):
    if variable not in model.params:
        return "—"

    return f"{model.params[variable]:.3f}{significance_stars(model.pvalues[variable])}"


def se_cell(model, variable):
    if variable not in model.params:
        return ""

    return f"({model.bse[variable]:.3f})"


daily_data = pd.read_csv(
    ANALYSIS_DATA_DIR / "daily_data.csv",
    parse_dates=["date"]
)

monthly_expiries = pd.read_csv(
    ANALYSIS_DATA_DIR / "monthly_expiries.csv",
    parse_dates=["expiry_date"]
)

monthly_expiries["expiry_id"] = monthly_expiries["expiry_id"].astype(object)

event_panel = pd.read_csv(
    ANALYSIS_DATA_DIR / "event_panel_14.csv",
    parse_dates=["date"]
)

event_panel["expiry_id"] = event_panel["expiry_id"].astype(object)

event_panel_wide = pd.read_csv(
    ANALYSIS_DATA_DIR / "event_panel_20.csv",
    parse_dates=["date"]
)

event_panel_wide["expiry_id"] = event_panel_wide["expiry_id"].astype(object)

ml_dataset = pd.read_csv(
    ANALYSIS_DATA_DIR / "ml_dataset.csv",
    parse_dates=["expiry_date"]
)

fomc_dates = pd.to_datetime(
    pd.read_csv(ANALYSIS_DATA_DIR / "fomc_dates.csv")["date"]
)

cpi_dates = pd.to_datetime(
    pd.read_csv(ANALYSIS_DATA_DIR / "cpi_dates.csv")["date"]
)

gdp_dates = pd.to_datetime(
    pd.read_csv(ANALYSIS_DATA_DIR / "gdp_dates.csv")["date"]
)

trading_calendar = (
    daily_data["date"]
    .sort_values()
    .values
    .astype("datetime64[D]")
)

event_window = 14
num_events = len(monthly_expiries)

newey_west = {"maxlags": 5}

market_controls = (
    "vix + sp500_ret + btc_ret + dxy_ret + dy10 + vvix + move + btc_realized_vol"
)

all_controls = (
    market_controls + " + fomc_dummy + cpi_dummy + gdp_dummy"
)

filtered_data = daily_data.dropna(
    subset=[
        "dxy_ret",
        "dy10",
        "vvix",
        "move",
        "btc_realized_vol",
    ]
)


reg1 = smf.ols(
    "btc_dvol ~ monthly_expiry + is_friday",
    data=daily_data
).fit(
    cov_type="HAC",
    cov_kwds=newey_west
)

reg2 = smf.ols(
    "btc_dvol ~ regular_monthly + quarterly_expiry + is_friday",
    data=daily_data
).fit(
    cov_type="HAC",
    cov_kwds=newey_west
)

reg3 = smf.ols(
    f"btc_dvol ~ monthly_expiry + is_friday + {all_controls}",
    data=filtered_data
).fit(
    cov_type="HAC",
    cov_kwds=newey_west
)

reg4 = smf.ols(
    f"btc_dvol ~ regular_monthly + quarterly_expiry + is_friday + {all_controls}",
    data=filtered_data
).fit(
    cov_type="HAC",
    cov_kwds=newey_west
)

reg5 = smf.ols(
    "btc_dvol ~ monthly_expiry + is_friday + vix + sp500_ret + btc_ret"
    " + fomc_dummy + cpi_dummy + gdp_dummy + dvol_lag",
    data=daily_data.dropna(subset=["dvol_lag"])
).fit(
    cov_type="HAC",
    cov_kwds=newey_west
)

dvol_changes_reg = smf.ols(
    "dvol_change ~ monthly_expiry + is_friday + vix + sp500_ret + btc_ret",
    data=daily_data.dropna(subset=["dvol_change"])
).fit(
    cov_type="HAC",
    cov_kwds=newey_west
)


a1_models = {
    "(1) Baseline": reg1,
    "(2) Quarterly split": reg2,
    "(3) Controls": reg3,
    "(4) Quarterly split + Full controls": reg4,
    "(5) Lagged DVOL": reg5,
}

a1_variables = [
    ("Monthly expiration", "monthly_expiry"),
    ("Regular Monthly expiration", "regular_monthly"),
    ("Quarterly expiration", "quarterly_expiry"),
    ("Friday indicator", "is_friday"),
    ("Lagged BTC-DVOL", "dvol_lag"),
]

a1_panel_a_rows = []

for label, variable in a1_variables:
    coef_row = {"Variable": label}
    se_row = {"Variable": ""}

    for column, model in a1_models.items():
        coef_row[column] = coef_cell(model, variable)
        se_row[column] = se_cell(model, variable)

    a1_panel_a_rows.append(coef_row)
    a1_panel_a_rows.append(se_row)


a1_panel_a_rows.append({
    "Variable": "Market controls",
    "(1) Baseline": "None",
    "(2) Quarterly split": "None",
    "(3) Controls": "Full",
    "(4) Quarterly split + Full controls": "Full",
    "(5) Lagged DVOL": "VIX, S&P 500, BTC Return",
})

a1_panel_a_rows.append({
    "Variable": "Macro controls",
    "(1) Baseline": "No",
    "(2) Quarterly split": "No",
    "(3) Controls": "Yes",
    "(4) Quarterly split + Full controls": "Yes",
    "(5) Lagged DVOL": "Yes",
})

a1_panel_a_rows.append({
    "Variable": "Observations",
    "(1) Baseline": f"{int(reg1.nobs):,}",
    "(2) Quarterly split": f"{int(reg2.nobs):,}",
    "(3) Controls": f"{int(reg3.nobs):,}",
    "(4) Quarterly split + Full controls": f"{int(reg4.nobs):,}",
    "(5) Lagged DVOL": f"{int(reg5.nobs):,}",
})

a1_panel_a_rows.append({
    "Variable": "R²",
    "(1) Baseline": f"{reg1.rsquared:.4f}",
    "(2) Quarterly split": f"{reg2.rsquared:.4f}",
    "(3) Controls": f"{reg3.rsquared:.4f}",
    "(4) Quarterly split + Full controls": f"{reg4.rsquared:.4f}",
    "(5) Lagged DVOL": f"{reg5.rsquared:.4f}",
})

pd.DataFrame(a1_panel_a_rows).to_csv(
    RESULTS_DIR / "TableA1_PanelA_Daily_Regressions.csv",
    index=False
)


a1_panel_b_rows = []

for label, variable in [
    ("Monthly expiration", "monthly_expiry"),
    ("Friday indicator", "is_friday"),
    ("VIX", "vix"),
    ("S&P 500 return", "sp500_ret"),
    ("Bitcoin return", "btc_ret"),
]:
    a1_panel_b_rows.append({
        "Variable": label,
        "Daily change in BTC-DVOL": coef_cell(dvol_changes_reg, variable),
    })

    a1_panel_b_rows.append({
        "Variable": "",
        "Daily change in BTC-DVOL": se_cell(dvol_changes_reg, variable),
    })


a1_panel_b_rows.append({
    "Variable": "Observations",
    "Daily change in BTC-DVOL": f"{int(dvol_changes_reg.nobs):,}",
})

a1_panel_b_rows.append({
    "Variable": "R²",
    "Daily change in BTC-DVOL": f"{dvol_changes_reg.rsquared:.4f}",
})

pd.DataFrame(a1_panel_b_rows).to_csv(
    RESULTS_DIR / "TableA1_PanelB_Daily_Change.csv",
    index=False
)


panel_main = smf.ols(
    "btc_dvol ~ dte + post + dte_post + vix + sp500_ret + btc_ret"
    " + fomc_dummy + cpi_dummy + gdp_dummy + C(expiry_id)",
    data=event_panel
).fit(
    cov_type="cluster",
    cov_kwds={"groups": event_panel["expiry_id"].values}
)

panel_quarterly = smf.ols(
    "btc_dvol ~ dte + post + dte_post + quarterly_post + quarterly_dte_post"
    " + vix + sp500_ret + btc_ret + fomc_dummy + cpi_dummy + gdp_dummy"
    " + C(expiry_id)",
    data=event_panel
).fit(
    cov_type="cluster",
    cov_kwds={"groups": event_panel["expiry_id"].values}
)

panel_hc3 = smf.ols(
    "btc_dvol ~ dte + post + dte_post + vix + sp500_ret + btc_ret"
    " + fomc_dummy + cpi_dummy + gdp_dummy + C(expiry_id)",
    data=event_panel
).fit(
    cov_type="HC3"
)

event_codes = pd.Categorical(event_panel["expiry_id"]).codes
date_codes = pd.Categorical(event_panel["date"]).codes

cluster_groups = np.column_stack([
    event_codes,
    date_codes
])

panel_twoway = smf.ols(
    "btc_dvol ~ dte + post + dte_post + vix + sp500_ret + btc_ret"
    " + fomc_dummy + cpi_dummy + gdp_dummy + C(expiry_id)",
    data=event_panel
).fit(
    cov_type="cluster",
    cov_kwds={"groups": cluster_groups}
)


a2_panel_a_rows = []

for label, variable in [
    ("Event time", "dte"),
    ("Post", "post"),
    ("Post × Event time", "dte_post"),
    ("Quarterly × Post", "quarterly_post"),
    ("Quarterly × Post × Event time", "quarterly_dte_post"),
]:
    a2_panel_a_rows.append({
        "Variable": label,
        "Estimate": coef_cell(panel_quarterly, variable),
    })

    a2_panel_a_rows.append({
        "Variable": "",
        "Estimate": se_cell(panel_quarterly, variable),
    })


a2_panel_a_rows.extend([
    {
        "Variable": "Market controls",
        "Estimate": "Yes",
    },
    {
        "Variable": "Macro controls",
        "Estimate": "Yes",
    },
    {
        "Variable": "Event fixed effects",
        "Estimate": "Yes",
    },
    {
        "Variable": "Observations",
        "Estimate": f"{int(panel_quarterly.nobs):,}",
    },
    {
        "Variable": "R²",
        "Estimate": f"{panel_quarterly.rsquared:.4f}",
    },
])

pd.DataFrame(a2_panel_a_rows).to_csv(
    RESULTS_DIR / "TableA2_PanelA_Quarterly_Expiration.csv",
    index=False
)


a2_panel_b_rows = []

for label, model in [
    ("HC3", panel_hc3),
    ("Clustered by expiration event", panel_main),
    ("Two-way clustered", panel_twoway),
]:
    a2_panel_b_rows.append({
        "Standard error method": label,
        "Post estimate": f"{model.params['post']:.3f}{significance_stars(model.pvalues['post'])}",
        "Standard error": f"{model.bse['post']:.3f}",
        "p-value": "<0.001" if model.pvalues["post"] < 0.001 else f"{model.pvalues['post']:.3f}",
    })

pd.DataFrame(a2_panel_b_rows).to_csv(
    RESULTS_DIR / "TableA2_PanelB_Alternative_SE.csv",
    index=False
)


bandwidth_rows = []

for bw_test in [5, 10, 14, 20]:
    bw_source = event_panel_wide if bw_test == 20 else event_panel

    bw_data = bw_source[
        bw_source.dte.abs() <= bw_test
    ].copy()

    bandwidth_model = smf.ols(
        "btc_dvol ~ dte + post + dte_post + vix + sp500_ret + btc_ret"
        " + fomc_dummy + cpi_dummy + gdp_dummy + C(expiry_id)",
        data=bw_data
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": bw_data["expiry_id"].values}
    )

    pval = bandwidth_model.pvalues["post"]

    if pval < 0.0001:
        p_display = "<0.001"
    else:
        p_display = f"{pval:.4f}"

    bandwidth_rows.append({
        "Event window": f"±{bw_test} trading days",
        "Expiration events": bw_data["expiry_id"].nunique(),
        "Post estimate": f"{bandwidth_model.params['post']:.3f}{significance_stars(pval)}",
        "p-value": p_display,
    })

pd.DataFrame(bandwidth_rows).to_csv(
    RESULTS_DIR / "TableA3_Event_Study_Windows.csv",
    index=False
)


midmonth_rows = []

for _, event in monthly_expiries.iterrows():

    yr2 = event["expiry_date"].year
    mo2 = event["expiry_date"].month

    fake_date = pd.Timestamp(
        year=yr2,
        month=mo2,
        day=15
    )

    while np.datetime64(fake_date, "D") not in trading_calendar:
        fake_date += pd.Timedelta(days=1)

    fake_index = np.searchsorted(
        trading_calendar,
        np.datetime64(fake_date, "D")
    )

    for offset in range(-event_window, event_window + 1):

        idx = fake_index + offset

        if idx < 0 or idx >= len(trading_calendar):
            continue

        actual_date = pd.Timestamp(
            trading_calendar[idx]
        )

        row = daily_data[
            daily_data.date == actual_date
        ]

        if len(row) == 0:
            continue

        r = row.iloc[0]

        midmonth_rows.append({
            "expiry_id": f"fake_{yr2}-{mo2:02d}",
            "dte": offset,
            "post": int(offset >= 0),
            "dte_post": offset * int(offset >= 0),
            "btc_dvol": r["btc_dvol"],
            "vix": r["vix"],
            "sp500_ret": r["sp500_ret"],
            "btc_ret": r["btc_ret"],
        })


midmonth_panel = pd.DataFrame(midmonth_rows)

midmonth_panel["expiry_id"] = (
    midmonth_panel["expiry_id"]
    .astype(object)
)

placebo_midmonth = smf.ols(
    "btc_dvol ~ dte + post + dte_post + vix + sp500_ret + btc_ret"
    " + C(expiry_id)",
    data=midmonth_panel
).fit(
    cov_type="cluster",
    cov_kwds={
        "groups": midmonth_panel["expiry_id"].values
    }
)


np.random.seed(42)

daily_data["event_expiry"] = (
    daily_data["date"]
    .isin(monthly_expiries.expiry_date)
    .astype(int)
)

other_fridays = daily_data[
    (daily_data.is_friday == 1)
    & (daily_data.monthly_expiry == 0)
]["date"].values

placebo_formula = (
    "btc_dvol ~ {expiry} + is_friday + vix + sp500_ret + btc_ret"
)

reference_model = smf.ols(
    placebo_formula.format(expiry="event_expiry"),
    data=daily_data
).fit()

reference_coef = reference_model.params[
    "event_expiry"
]

n_draws = 1000
draw_coefs = []

for i in range(n_draws):

    draw_dates = pd.Series(
        other_fridays
    ).sample(
        len(monthly_expiries),
        random_state=i
    )

    daily_data["draw_expiry"] = (
        daily_data["date"]
        .isin(draw_dates)
        .astype(int)
    )

    draw_model = smf.ols(
        placebo_formula.format(
            expiry="draw_expiry"
        ),
        data=daily_data
    ).fit()

    draw_coefs.append(
        draw_model.params["draw_expiry"]
    )


draw_coefs = np.array(draw_coefs)

p_one_side = (
    1 + np.sum(draw_coefs <= reference_coef)
) / (
    n_draws + 1
)

p_two_side = min(
    1,
    2 * min(
        p_one_side,
        1 - p_one_side
    )
)

draw_pct = (
    np.mean(
        draw_coefs <= reference_coef
    )
    * 100
)


same_month_coefs = []

for seed in range(n_draws):

    rng = np.random.RandomState(seed)

    matched_dates = []

    for _, event in monthly_expiries.iterrows():

        yr = event["expiry_date"].year
        mo = event["expiry_date"].month

        same_month = daily_data[
            (daily_data.is_friday == 1)
            & (daily_data.monthly_expiry == 0)
            & (daily_data.date.dt.year == yr)
            & (daily_data.date.dt.month == mo)
        ]["date"].values

        if len(same_month) > 0:
            matched_dates.append(
                rng.choice(same_month)
            )

    if len(matched_dates) == len(monthly_expiries):

        daily_data["mm_expiry"] = (
            daily_data["date"]
            .isin(matched_dates)
            .astype(int)
        )

        mm_model = smf.ols(
            placebo_formula.format(
                expiry="mm_expiry"
            ),
            data=daily_data
        ).fit()

        same_month_coefs.append(
            mm_model.params["mm_expiry"]
        )


same_month_coefs = np.array(
    same_month_coefs
)

p_one_side_mm = (
    1
    + np.sum(
        same_month_coefs <= reference_coef
    )
) / (
    len(same_month_coefs) + 1
)

p_two_side_mm = min(
    1,
    2 * min(
        p_one_side_mm,
        1 - p_one_side_mm
    )
)

draw_pct_mm = (
    np.mean(
        same_month_coefs <= reference_coef
    )
    * 100
)


a4_panel_a_rows = [
    {
        "Placebo test": "Mid-month placebo",
        "Estimate": f"{placebo_midmonth.params['post']:.3f}",
        "Share at least as negative": "—",
        "One-sided p-value": "—",
        "Two-sided p-value": f"{placebo_midmonth.pvalues['post']:.3f}",
    },
    {
        "Placebo test": "Other Fridays, 1,000 draws",
        "Estimate": f"{reference_coef:.3f}",
        "Share at least as negative": f"{draw_pct:.1f}%",
        "One-sided p-value": f"{p_one_side:.3f}",
        "Two-sided p-value": f"{p_two_side:.3f}",
    },
    {
        "Placebo test": "Same-month Fridays, 1,000 draws",
        "Estimate": f"{reference_coef:.3f}",
        "Share at least as negative": f"{draw_pct_mm:.1f}%",
        "One-sided p-value": f"{p_one_side_mm:.3f}",
        "Two-sided p-value": f"{p_two_side_mm:.3f}",
    },
]

pd.DataFrame(a4_panel_a_rows).to_csv(
    RESULTS_DIR / "TableA4_PanelA_Placebo_Tests.csv",
    index=False
)


fomc_window = set(
    (d + pd.Timedelta(days=delta)).date()
    for d in fomc_dates
    for delta in [-1, 0, 1]
)

cpi_window = set(
    (d + pd.Timedelta(days=delta)).date()
    for d in cpi_dates
    for delta in [-1, 0, 1]
)

gdp_window = set(
    (d + pd.Timedelta(days=delta)).date()
    for d in gdp_dates
    for delta in [-1, 0, 1]
)

all_macro = (
    fomc_window
    | cpi_window
    | gdp_window
)


macro_rows = []

for label, window in [
    ("FOMC", fomc_window),
    ("CPI release", cpi_window),
    ("GDP release", gdp_window),
    ("All macro announcements", all_macro),
]:

    excluded_events = monthly_expiries[
        ~monthly_expiries["expiry_date"]
        .dt.date
        .isin(window)
    ].copy()

    num_dropped = (
        len(monthly_expiries)
        - len(excluded_events)
    )

    excluded_panel = event_panel[
        event_panel.expiry_id.isin(
            excluded_events.expiry_id
        )
    ].copy()

    exclusion_model = smf.ols(
        "btc_dvol ~ dte + post + dte_post + vix + sp500_ret + btc_ret"
        " + fomc_dummy + cpi_dummy + gdp_dummy + C(expiry_id)",
        data=excluded_panel
    ).fit(
        cov_type="cluster",
        cov_kwds={
            "groups":
                excluded_panel["expiry_id"].values
        }
    )

    pval = exclusion_model.pvalues["post"]

    macro_rows.append({
        "Exclusion": label,
        "Expiration Events excluded": num_dropped,
        "Expiration Events remaining": len(excluded_events),
        "Post estimate":
            f"{exclusion_model.params['post']:.3f}"
            f"{significance_stars(pval)}",
        "p-value":
            "<0.001"
            if pval < 0.001
            else f"{pval:.3f}",
    })


pd.DataFrame(macro_rows).to_csv(
    RESULTS_DIR / "TableA4_PanelB_Macro_Exclusions.csv",
    index=False
)


feature_names = [
    "dvol_level_5d",
    "dvol_change_5d",
    "dvol_change_10d",
    "dvol_std_10d",
    "dvol_vix_spread",
    "vix_level_5d",
    "vix_change_5d",
    "btc_rvol_5d",
    "btc_ret_cum_5d",
    "sp500_ret_5d",
    "dxy_ret_5d",
    "vvix_5d",
    "move_5d",
    "is_quarterly",
]

target_name = "abnormal_dvol"

features = ml_dataset[
    feature_names
].values

outcome = ml_dataset[
    target_name
].values


random_forest = RandomForestRegressor(
    n_estimators=500,
    max_depth=3,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)

gradient_boost = GradientBoostingRegressor(
    n_estimators=100,
    max_depth=2,
    learning_rate=0.05,
    random_state=42
)

lasso_pipe = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "lasso",
        Lasso(
            alpha=0.5,
            max_iter=5000,
            random_state=42
        )
    ),
])


pred_rf = run_leave_one_out(
    random_forest,
    features,
    outcome
)

pred_gb = run_leave_one_out(
    gradient_boost,
    features,
    outcome
)

pred_lasso = run_leave_one_out(
    lasso_pipe,
    features,
    outcome
)


loocv_obj = LeaveOneOut()

pred_naive = np.zeros(
    len(outcome)
)

for train_idx, test_idx in loocv_obj.split(
    features
):
    pred_naive[test_idx] = (
        outcome[train_idx].mean()
    )


c1_panel_a_rows = [
    {
        "Model": "Simple average",
        "RMSE":
            f"{np.sqrt(mean_squared_error(outcome, pred_naive)):.3f}",
        "Cross-validated R²":
            f"{r2_score(outcome, pred_naive):.3f}",
        "Prediction correlation": "—",
    },
]

for label, predicted in [
    ("Random Forest", pred_rf),
    ("Gradient Boosting", pred_gb),
    ("LASSO", pred_lasso),
]:

    corr = scipy_stats.pearsonr(
        outcome,
        predicted
    )[0]

    c1_panel_a_rows.append({
        "Model": label,
        "RMSE":
            f"{np.sqrt(mean_squared_error(outcome, predicted)):.3f}",
        "Cross-validated R²":
            f"{r2_score(outcome, predicted):.3f}",
        "Prediction correlation":
            f"{corr:.3f}",
    })


pd.DataFrame(c1_panel_a_rows).to_csv(
    RESULTS_DIR / "TableC1_PanelA_ML_Performance.csv",
    index=False
)


random_forest.fit(
    features,
    outcome
)

feature_importance = pd.Series(
    random_forest.feature_importances_,
    index=feature_names
).sort_values(
    ascending=False
)


FEATURE_LABELS = {
    "dvol_level_5d": "5-day average DVOL level",
    "dvol_change_5d": "5-day DVOL change",
    "dvol_change_10d": "10-day DVOL change",
    "dvol_std_10d": "10-day DVOL standard deviation",
    "dvol_vix_spread": "DVOL minus VIX spread",
    "vix_level_5d": "5-day average VIX",
    "vix_change_5d": "5-day VIX change",
    "btc_rvol_5d": "5-day BTC realized volatility",
    "btc_ret_cum_5d": "5-day cumulative BTC return",
    "sp500_ret_5d": "5-day cumulative S&P 500 return",
    "dxy_ret_5d": "5-day cumulative DXY return",
    "vvix_5d": "5-day average VVIX",
    "move_5d": "5-day average MOVE index",
    "is_quarterly": "Quarterly expiration indicator",
}


c1_panel_b_rows = []

for feature, importance in feature_importance.items():

    corr = np.corrcoef(
        ml_dataset[feature],
        ml_dataset[target_name]
    )[0, 1]

    c1_panel_b_rows.append({
        "Predictor":
            FEATURE_LABELS.get(
                feature,
                feature
            ),
        "Importance":
            f"{importance:.3f}",
        "Correlation with Abnormal DVOL":
            f"{corr:.3f}",
    })


pd.DataFrame(c1_panel_b_rows).to_csv(
    RESULTS_DIR / "TableC1_PanelB_RF_Importance.csv",
    index=False
)


ols_features = [
    f
    for f in feature_names
    if f != "dvol_vix_spread"
]

ols_confirmation = smf.ols(
    f"{target_name} ~ "
    + " + ".join(ols_features),
    data=ml_dataset[
        feature_names
        + [target_name]
    ]
).fit(
    cov_type="HC3"
)


c2_rows = []

for variable in ols_features:

    c2_rows.append({
        "Predictor":
            FEATURE_LABELS.get(
                variable,
                variable
            ),
        "Expiration-Day Abnormal BTC-DVOL":
            f"{ols_confirmation.params[variable]:.3f}"
            f"{significance_stars(ols_confirmation.pvalues[variable])}",
    })

    c2_rows.append({
        "Predictor": "",
        "Expiration-Day Abnormal BTC-DVOL":
            f"({ols_confirmation.bse[variable]:.3f})",
    })


c2_rows.extend([
    {
        "Predictor": "Observations",
        "Expiration-Day Abnormal BTC-DVOL":
            f"{int(ols_confirmation.nobs):,}",
    },
    {
        "Predictor": "R²",
        "Expiration-Day Abnormal BTC-DVOL":
            f"{ols_confirmation.rsquared:.4f}",
    },
    {
        "Predictor": "Adjusted R²",
        "Expiration-Day Abnormal BTC-DVOL":
            f"{ols_confirmation.rsquared_adj:.4f}",
    },
])


pd.DataFrame(c2_rows).to_csv(
    RESULTS_DIR / "TableC2_OLS_Confirmation.csv",
    index=False
)


FONT_AXIS = 12
FONT_TICK = 11
FONT_ANNOT = 10

fig_b1, ax_b1 = plt.subplots(
    figsize=(8, 7)
)

dot_colors = [
    "#C62828"
    if q
    else "#1F4E79"
    for q in ml_dataset["is_quarterly"]
]

ax_b1.scatter(
    outcome,
    pred_gb,
    c=dot_colors,
    alpha=0.8,
    s=60,
    zorder=3
)

plot_limits = [
    min(
        outcome.min(),
        pred_gb.min()
    ) - 1,
    max(
        outcome.max(),
        pred_gb.max()
    ) + 1,
]

ax_b1.plot(
    plot_limits,
    plot_limits,
    "k--",
    lw=1,
    alpha=0.5
)

ax_b1.text(
    0.05,
    0.93,
    f"LOOCV $R^2$ = {r2_score(outcome, pred_gb):.3f}",
    transform=ax_b1.transAxes,
    fontsize=FONT_ANNOT,
    color="#1F4E79",
    bbox=dict(
        boxstyle="round,pad=0.3",
        facecolor="#F5F7FA",
        edgecolor="#E0E8F0"
    )
)

ax_b1.text(
    plot_limits[1] - 0.5,
    plot_limits[1] - 1,
    "Monthly",
    color="#1F4E79",
    fontsize=FONT_ANNOT,
    ha="right"
)

ax_b1.text(
    plot_limits[1] - 0.5,
    plot_limits[1] - 2.5,
    "Quarterly",
    color="#C62828",
    fontsize=FONT_ANNOT,
    ha="right"
)

ax_b1.set_xlabel(
    "Expiration-day abnormal BTC-DVOL (index points)",
    fontsize=FONT_AXIS
)

ax_b1.set_ylabel(
    "Gradient Boosting prediction (LOOCV)",
    fontsize=FONT_AXIS
)

ax_b1.tick_params(
    labelsize=FONT_TICK
)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "FigureB1_Predicted_vs_Actual.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()


assert len(daily_data) == 1003
assert len(monthly_expiries) == 46
assert len(event_panel) == 1334
assert len(event_panel_wide) == 1845
assert len(ml_dataset) == 46

assert np.isclose(
    reg2.params["regular_monthly"],
    -2.2770,
    atol=0.002
)

assert np.isclose(
    reg2.params["quarterly_expiry"],
    -2.8955,
    atol=0.002
)

assert np.isclose(
    reg4.params["regular_monthly"],
    -2.2735,
    atol=0.002
)

assert np.isclose(
    reg4.params["quarterly_expiry"],
    1.4484,
    atol=0.002
)

assert np.isclose(
    dvol_changes_reg.params["monthly_expiry"],
    -1.5387,
    atol=0.002
)

assert np.isclose(
    panel_quarterly.params["post"],
    -3.8222,
    atol=0.002
)

assert np.isclose(
    panel_quarterly.params["quarterly_post"],
    0.2138,
    atol=0.002
)

assert np.isclose(
    panel_main.bse["post"],
    0.8827,
    atol=0.002
)

assert np.isclose(
    panel_twoway.bse["post"],
    0.9251,
    atol=0.003
)

assert np.isclose(
    placebo_midmonth.params["post"],
    1.3161,
    atol=0.002
)

assert np.isclose(
    reference_coef,
    -2.1865,
    atol=0.002
)

assert np.isclose(
    p_two_side,
    0.1339,
    atol=0.002
)

assert np.isclose(
    p_two_side_mm,
    0.0579,
    atol=0.002
)

assert np.isclose(
    r2_score(outcome, pred_gb),
    0.6727,
    atol=0.003
)

assert np.isclose(
    ols_confirmation.params["dvol_change_10d"],
    0.2643,
    atol=0.003
)

print("Appendix analysis completed successfully.")
print(f"Daily-change expiration coefficient: {dvol_changes_reg.params['monthly_expiry']:.4f}")
print(f"Quarterly-model regular expiration post coefficient: {panel_quarterly.params['post']:.4f}")
print(f"Event-clustered post SE: {panel_main.bse['post']:.4f}")
print(f"Two-way clustered post SE: {panel_twoway.bse['post']:.4f}")
print(f"Mid-month placebo: {placebo_midmonth.params['post']:.4f}, p={placebo_midmonth.pvalues['post']:.4f}")
print(f"Other-Friday two-sided p-value: {p_two_side:.4f}")
print(f"Same-month-Friday two-sided p-value: {p_two_side_mm:.4f}")
print(f"Gradient Boosting LOOCV R²: {r2_score(outcome, pred_gb):.4f}")
print(f"OLS 10-day DVOL-change coefficient: {ols_confirmation.params['dvol_change_10d']:.4f}")
print("Appendix Tables A1-A4, Figure B1, and Tables C1-C2 saved to the results folder.")