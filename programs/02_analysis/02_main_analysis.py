import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.ensemble import RandomForestRegressor

PROGRAMS_DIR = Path(__file__).resolve().parents[1]
if str(PROGRAMS_DIR) not in sys.path:
    sys.path.insert(0, str(PROGRAMS_DIR))

from config import ANALYSIS_DATA_DIR, RESULTS_DIR


def significance_stars(pvalue):
    return "***" if pvalue < 0.01 else "**" if pvalue < 0.05 else "*" if pvalue < 0.1 else ""


daily_data = pd.read_csv(
    ANALYSIS_DATA_DIR / "daily_data.csv",
    parse_dates=["date"]
)

event_panel = pd.read_csv(
    ANALYSIS_DATA_DIR / "event_panel_14.csv",
    parse_dates=["date"]
)

event_panel["expiry_id"] = event_panel["expiry_id"].astype(object)

ml_dataset = pd.read_csv(
    ANALYSIS_DATA_DIR / "ml_dataset.csv",
    parse_dates=["expiry_date"]
)


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

reg3 = smf.ols(
    f"btc_dvol ~ monthly_expiry + is_friday + {all_controls}",
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


panel_nocontrols = smf.ols(
    "btc_dvol ~ dte + post + dte_post + C(expiry_id)",
    data=event_panel
).fit(
    cov_type="cluster",
    cov_kwds={"groups": event_panel["expiry_id"].values}
)

panel_main = smf.ols(
    "btc_dvol ~ dte + post + dte_post + vix + sp500_ret + btc_ret"
    " + fomc_dummy + cpi_dummy + gdp_dummy + C(expiry_id)",
    data=event_panel
).fit(
    cov_type="cluster",
    cov_kwds={"groups": event_panel["expiry_id"].values}
)


stat_vars = {
    "btc_dvol": "BTC-DVOL (index points)",
    "vix": "VIX (index points)",
    "sp500_ret": "S&P 500 return (%)",
    "btc_ret": "Bitcoin return (%)",
    "dxy_ret": "DXY return (%)",
    "dy10": "10-year Treasury yield change (pp)",
    "vvix": "VVIX (index points)",
    "move": "MOVE index",
    "btc_realized_vol": "7-day realized Bitcoin volatility (%)",
    "monthly_expiry": "Monthly expiration indicator",
    "quarterly_expiry": "Quarterly expiration indicator",
    "fomc_dummy": "FOMC announcement window",
    "cpi_dummy": "CPI announcement window",
    "gdp_dummy": "GDP announcement window",
}

stat_rows = []

for var, label in stat_vars.items():
    if var in daily_data.columns:
        col = daily_data[var].dropna()

        stat_rows.append({
            "Variable": label,
            "Mean": round(col.mean(), 3),
            "Std. Dev.": round(col.std(), 3),
            "Min": round(col.min(), 3),
            "Max": round(col.max(), 3),
            "N": len(col),
        })

table1 = pd.DataFrame(stat_rows)

table1.to_csv(
    RESULTS_DIR / "Table01_Summary_Statistics.csv",
    index=False
)


def coefficient_cell(model, variable):
    if variable not in model.params:
        return "—"

    coef = model.params[variable]
    pval = model.pvalues[variable]

    return f"{coef:.3f}{significance_stars(pval)}"


def se_cell(model, variable):
    if variable not in model.params:
        return ""

    return f"({model.bse[variable]:.3f})"


panel_a_models = {
    "(1) Baseline": reg1,
    "(2) Controls": reg3,
    "(3) Lagged DVOL": reg5,
}

panel_a_variables = [
    ("Monthly Expirations", "monthly_expiry"),
    ("Friday Indicator", "is_friday"),
    ("Lagged BTC-DVOL", "dvol_lag"),
]

panel_a_rows = []

for label, variable in panel_a_variables:
    coef_row = {"Variable": label}
    se_row = {"Variable": ""}

    for column, model in panel_a_models.items():
        coef_row[column] = coefficient_cell(model, variable)
        se_row[column] = se_cell(model, variable)

    panel_a_rows.append(coef_row)
    panel_a_rows.append(se_row)


panel_a_rows.append({
    "Variable": "Market Controls",
    "(1) Baseline": "None",
    "(2) Controls": "Full",
    "(3) Lagged DVOL": "VIX, S&P 500, BTC Return",
})

panel_a_rows.append({
    "Variable": "Macro Controls",
    "(1) Baseline": "No",
    "(2) Controls": "Yes",
    "(3) Lagged DVOL": "Yes",
})

panel_a_rows.append({
    "Variable": "Observations",
    "(1) Baseline": f"{int(reg1.nobs):,}",
    "(2) Controls": f"{int(reg3.nobs):,}",
    "(3) Lagged DVOL": f"{int(reg5.nobs):,}",
})

panel_a_rows.append({
    "Variable": "R²",
    "(1) Baseline": f"{reg1.rsquared:.4f}",
    "(2) Controls": f"{reg3.rsquared:.4f}",
    "(3) Lagged DVOL": f"{reg5.rsquared:.4f}",
})

table2_panel_a = pd.DataFrame(panel_a_rows)

table2_panel_a.to_csv(
    RESULTS_DIR / "Table02_PanelA_Daily_Regressions.csv",
    index=False
)


panel_b_models = {
    "(1) Baseline": panel_nocontrols,
    "(2) Controls": panel_main,
}

panel_b_variables = [
    ("Post", "post"),
    ("Event time", "dte"),
    ("Post × Event time", "dte_post"),
    ("VIX", "vix"),
    ("S&P 500 return", "sp500_ret"),
    ("Bitcoin return", "btc_ret"),
    ("FOMC window", "fomc_dummy"),
    ("CPI window", "cpi_dummy"),
    ("GDP window", "gdp_dummy"),
]

panel_b_rows = []

for label, variable in panel_b_variables:
    coef_row = {"Variable": label}
    se_row = {"Variable": ""}

    for column, model in panel_b_models.items():
        coef_row[column] = coefficient_cell(model, variable)
        se_row[column] = se_cell(model, variable)

    panel_b_rows.append(coef_row)
    panel_b_rows.append(se_row)


panel_b_rows.append({
    "Variable": "Event fixed effects",
    "(1) Baseline": "Yes",
    "(2) Controls": "Yes",
})

panel_b_rows.append({
    "Variable": "Observations",
    "(1) Baseline": f"{int(panel_nocontrols.nobs):,}",
    "(2) Controls": f"{int(panel_main.nobs):,}",
})

panel_b_rows.append({
    "Variable": "R²",
    "(1) Baseline": f"{panel_nocontrols.rsquared:.4f}",
    "(2) Controls": f"{panel_main.rsquared:.4f}",
})

table2_panel_b = pd.DataFrame(panel_b_rows)

table2_panel_b.to_csv(
    RESULTS_DIR / "Table02_PanelB_Event_Study.csv",
    index=False
)


FONT_AXIS = 12
FONT_TICK = 11
FONT_ANNOT = 10


fig1, ax1 = plt.subplots(figsize=(10, 6))

dte_avg = event_panel.groupby("dte")["btc_dvol"].agg(
    ["mean", "sem"]
)

pre_mean = event_panel[
    event_panel.dte < 0
]["btc_dvol"].mean()

ax1.plot(
    dte_avg.index,
    dte_avg["mean"],
    color="#1F4E79",
    lw=2
)

ax1.errorbar(
    dte_avg.index,
    dte_avg["mean"],
    yerr=1.96 * dte_avg["sem"],
    fmt="none",
    color="#1F4E79",
    alpha=0.5,
    capsize=3,
    lw=1
)

ax1.axvline(
    0,
    color="red",
    lw=1.5,
    ls="--"
)

ax1.axhline(
    pre_mean,
    color="gray",
    lw=1,
    ls=":"
)

ax1.text(
    0.5,
    pre_mean + 0.3,
    f"Pre-expiration mean = {pre_mean:.1f}",
    fontsize=FONT_ANNOT,
    color="gray",
    ha="center",
    transform=ax1.get_yaxis_transform()
)

ax1.text(
    0.02,
    0.97,
    "Expiration day",
    color="red",
    fontsize=FONT_ANNOT,
    transform=ax1.transAxes,
    va="top"
)

ax1.set_xlabel(
    "Event time (U.S. trading days relative to expiration)",
    fontsize=FONT_AXIS
)

ax1.set_ylabel(
    "BTC-DVOL (index points)",
    fontsize=FONT_AXIS
)

ax1.tick_params(
    labelsize=FONT_TICK
)

ax1.set_xticks(
    range(-14, 15, 2)
)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "Figure01_Average_DVOL_Levels.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()


fig2, ax2 = plt.subplots(figsize=(10, 6))

abn_avg = event_panel.groupby(
    "dte"
)["abnormal_dvol"].agg(
    ["mean", "sem"]
)

bar_clrs = [
    "#C62828" if v < 0 else "#2E75B6"
    for v in abn_avg["mean"]
]

ax2.bar(
    abn_avg.index,
    abn_avg["mean"],
    color=bar_clrs,
    alpha=0.75,
    width=0.8
)

ax2.errorbar(
    abn_avg.index,
    abn_avg["mean"],
    yerr=1.96 * abn_avg["sem"],
    fmt="none",
    color="black",
    capsize=3,
    lw=1
)

ax2.axhline(
    0,
    color="black",
    lw=0.8
)

ax2.axvline(
    0,
    color="red",
    lw=1.5,
    ls="--"
)

ax2.text(
    0.02,
    0.97,
    "Expiration day",
    color="red",
    fontsize=FONT_ANNOT,
    transform=ax2.transAxes,
    va="top"
)

ax2.set_xlabel(
    "Event time (U.S. trading days relative to expiration)",
    fontsize=FONT_AXIS
)

ax2.set_ylabel(
    "Abnormal BTC-DVOL (index points)",
    fontsize=FONT_AXIS
)

ax2.tick_params(
    labelsize=FONT_TICK
)

ax2.set_xticks(
    range(-14, 15, 2)
)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "Figure02_Abnormal_DVOL_by_DTE.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()


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
    "dvol_level_5d": "5-day avg. DVOL level",
    "dvol_change_5d": "5-day DVOL change",
    "dvol_change_10d": "10-day DVOL change",
    "dvol_std_10d": "10-day DVOL std. dev.",
    "dvol_vix_spread": "DVOL minus VIX spread",
    "vix_level_5d": "5-day avg. VIX",
    "vix_change_5d": "5-day VIX change",
    "btc_rvol_5d": "5-day BTC realized vol.",
    "btc_ret_cum_5d": "5-day cumulative BTC return",
    "sp500_ret_5d": "5-day cumulative S&P 500 return",
    "dxy_ret_5d": "5-day cumulative DXY return",
    "vvix_5d": "5-day avg. VVIX",
    "move_5d": "5-day avg. MOVE index",
    "is_quarterly": "Quarterly expiration indicator",
}


feature_importance_table = pd.DataFrame({
    "Feature": [
        FEATURE_LABELS.get(f, f)
        for f in feature_importance.index
    ],

    "Importance":
        feature_importance.values,

    "Correlation_with_abnormal_DVOL": [
        np.corrcoef(
            ml_dataset[f],
            ml_dataset[target_name]
        )[0, 1]
        for f in feature_importance.index
    ],
})

feature_importance_table.to_csv(
    RESULTS_DIR / "Figure03_Feature_Importance_Data.csv",
    index=False
)


fig3, ax3 = plt.subplots(
    figsize=(10, 7)
)

feat_sorted = feature_importance.sort_values(
    ascending=True
)

feat_labels = [
    FEATURE_LABELS.get(f, f)
    for f in feat_sorted.index
]

feat_colors = [
    "#C62828"
    if np.corrcoef(
        ml_dataset[f],
        ml_dataset[target_name]
    )[0, 1] > 0
    else "#2E75B6"
    for f in feat_sorted.index
]

ax3.barh(
    range(len(feat_sorted)),
    feat_sorted.values,
    color=feat_colors,
    alpha=0.8
)

ax3.set_yticks(
    range(len(feat_sorted))
)

ax3.set_yticklabels(
    feat_labels,
    fontsize=FONT_TICK
)

ax3.set_xlabel(
    "Random Forest feature importance",
    fontsize=FONT_AXIS
)

ax3.text(
    0.5,
    -0.13,
    "Bar length = feature importance. Red: positive correlation with expiration-day abnormal DVOL. Blue: negative correlation.",
    transform=ax3.transAxes,
    fontsize=9,
    ha="center",
    va="top",
    style="italic",
    color="#444444",
    wrap=True
)

ax3.tick_params(
    labelsize=FONT_TICK
)

ax3.text(
    0.98,
    0.06,
    "Higher value → larger expiration-day drop",
    color="#2E75B6",
    fontsize=FONT_ANNOT - 1,
    ha="right",
    va="bottom",
    transform=ax3.transAxes
)

ax3.text(
    0.98,
    0.12,
    "Higher value → smaller expiration-day drop",
    color="#C62828",
    fontsize=FONT_ANNOT - 1,
    ha="right",
    va="bottom",
    transform=ax3.transAxes
)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "Figure03_Feature_Importance.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()


assert len(daily_data) == 1003
assert len(event_panel) == 1334
assert event_panel["expiry_id"].nunique() == 46
assert len(ml_dataset) == 46

assert np.isclose(
    reg1.params["monthly_expiry"],
    -2.4744,
    atol=0.001
)

assert np.isclose(
    reg3.params["monthly_expiry"],
    -1.0661,
    atol=0.001
)

assert np.isclose(
    reg5.params["monthly_expiry"],
    -1.2414,
    atol=0.001
)

assert np.isclose(
    panel_nocontrols.params["post"],
    -3.4976,
    atol=0.001
)

assert np.isclose(
    panel_main.params["post"],
    -3.7188,
    atol=0.001
)

assert np.isclose(
    feature_importance.iloc[0],
    0.640179,
    atol=0.001
)

assert feature_importance.index[0] == "dvol_change_10d"

print("Main analysis completed successfully.")
print(f"Baseline monthly expiration coefficient: {reg1.params['monthly_expiry']:.4f}")
print(f"Controls monthly expiration coefficient: {reg3.params['monthly_expiry']:.4f}")
print(f"Lagged-DVOL monthly expiration coefficient: {reg5.params['monthly_expiry']:.4f}")
print(f"Baseline event-study post coefficient: {panel_nocontrols.params['post']:.4f}")
print(f"Main event-study post coefficient: {panel_main.params['post']:.4f}")
print(f"Main event-study post p-value: {panel_main.pvalues['post']:.4f}")
print(f"Top Random Forest predictor: {feature_importance.index[0]}")
print(f"Top feature importance: {feature_importance.iloc[0]:.4f}")
print("Table 1, Table 2, and Figures 1-3 saved to the results folder.")