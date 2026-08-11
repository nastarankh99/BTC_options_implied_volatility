# Bitcoin Option Expirations and Implied Volatility

Replication package for:

Nastaran Khorram  
ECON 899 MA Paper  
August 2026

## Overview

This repository contains the code used to reproduce the results in the paper *Bitcoin Option Expirations and Implied Volatility*.

The paper studies whether monthly Bitcoin option expirations on Deribit are associated with changes in BTC-DVOL, Deribit's 30-day Bitcoin implied volatility index. It also studies whether information available before expiration helps predict the size of the expiration-day movement.

The daily sample runs from January 3, 2022 to December 31, 2025. The main event-study sample contains 46 monthly expiration events from January 2022 to November 2025.

## Software

The replication package uses Python 3.11.

The required Python packages and versions are listed in `requirements.txt`.

To install the required packages, run:

```bash
python programs/00_setup.py
```

## Raw Data

The raw data are not included in this repository because of data redistribution restrictions.

Before running the replication, place the following three files in `data/raw_data/` using these exact filenames:

- `btc_dvol_complete_2021_2025.csv`  
- `bloomberg_data_session4_extended_2025_december.xlsx`
- `bloomberg_data_session7_final.xlsx`

### Deribit BTC-DVOL Data

BTC-DVOL data were obtained from the Deribit API v2 using the `public/get_volatility_index_data` endpoint:

`https://www.deribit.com/api/v2/public/get_volatility_index_data`

The analysis uses BTC observations and the daily closing value of DVOL. The file should be saved as:

`data/raw_data/btc_dvol_complete_2021_2025.csv`
(The CSV file used in the analysis contains the historical daily BTC-DVOL observations obtained from the Deribit API for 2022–2025.)
The raw Deribit data are not redistributed in this repository.

### Bloomberg Financial Market Data

Financial market data were obtained from the Bloomberg Terminal. The file should be saved as:

`data/raw_data/bloomberg_data_session4_extended_2025_december.xlsx`

The workbook contains the following sheets:

- `VIX`
- `S&P 500`
- `Bitcoin spot USD`
- `US Dollar Index`
- `US 10yr Yield`
- `VVIX`
- `MOVE`

The analysis constructs log returns for the S&P 500, Bitcoin spot price, and the U.S. Dollar Index. It also calculates the daily change in the U.S. 10-year Treasury yield.

Bitcoin returns are calculated from consecutive Bloomberg Bitcoin spot observations before the data are matched to the U.S. trading-day sample. Because Bitcoin trades continuously, these returns may span calendar days rather than U.S. trading days.

### Bloomberg Macroeconomic Data

The dates of FOMC rate decisions, CPI releases, and GDP releases were also obtained from the Bloomberg Terminal. The file should be saved as:

`data/raw_data/bloomberg_data_session7_final.xlsx`

The workbook contains the following sheets and series:

- `FOMC`: `FOMC Rate Decision (Upper Bound)`
- `CPI`: `CPI MoM`
- `GDP `: `GDP Annualized QoQ` 

The analysis creates indicators for U.S. trading days that fall within one calendar day of each announcement.

## Running the Replication

After placing the three raw data files in `data/raw_data/`, install the required packages if needed:

```bash
python programs/00_setup.py
```

Then run the full replication with:

```bash
python programs/01_main.py
```

The master program runs the data preparation, main analysis, and appendix analysis in order.

The generated analysis datasets are saved in:

`data/data_for_analysis/`

The generated tables and figures are saved in:

`results/`

These generated files are not stored in the GitHub repository.

## Program Structure

`programs/config.py` contains the project paths and raw-data filenames.

`programs/00_setup.py` installs the required Python packages.

`programs/01_main.py` is the master program that runs the complete replication.

`programs/01_dataprep/01_prepare_analysis_data.py` prepares the daily dataset, expiration-event samples, event-study panels, and machine-learning dataset.

`programs/02_analysis/02_main_analysis.py` reproduces Table 1, Table 2, and Figures 1-3 in the main paper.

`programs/03_appendix/03_appendix.py` reproduces Appendix Tables A1-A4, Figure B1, and Tables C1-C2.

## Main Outputs

The main analysis produces:

- Table 1: Summary Statistics for the Daily Estimation Sample
- Table 2: Daily Regression and Event Study Estimates of BTC-DVOL Around Monthly Option Expiration
- Figure 1: Average BTC-DVOL Levels Around Monthly Option Expiration
- Figure 2: Average Abnormal BTC-DVOL Around Monthly Option Expiration
- Figure 3: Random Forest Feature Importance for Expiration-Day Abnormal BTC-DVOL

The appendix analysis produces:

- Tables A1-A4
- Figure B1
- Tables C1-C2

## Reproducibility Checks

The programs include checks for the main sample sizes and results.

The expected samples are:

- 1,003 U.S. trading days
- 46 monthly expiration events
- 32 regular monthly expirations
- 14 quarterly expirations
- 1,334 observations in the main ±14 trading-day event-study panel
- 1,845 observations in the complete ±20 trading-day panel
- 46 observations in the machine-learning sample

The main event-study estimate for the Post coefficient is approximately -3.72 BTC-DVOL points.

Small numerical differences in the tree-based machine-learning results may occur across software environments. These small differences do not affect the conclusions of the analysis.

## Data Sources

Bloomberg L.P. 2026. *Bloomberg Terminal*. Financial market and macroeconomic data.

Deribit. 2026. `public/get_volatility_index_data`. Deribit API v2.

## Contact

Nastaran Khorram  
Simon Fraser University (nka113@sfu.ca)