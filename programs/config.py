from pathlib import Path


# ECON 899 Replication Package
# Bitcoin Option Expirations and Implied Volatility
#
# This is the only configuration file that a replicator
# may need to edit.


# Repository root
ROOT = Path(__file__).resolve().parent.parent

# Main directories
DATA_DIR = ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw_data"
ANALYSIS_DATA_DIR = DATA_DIR / "data_for_analysis"
RESULTS_DIR = ROOT / "results"
PROGRAMS_DIR = ROOT / "programs"

# Raw input files
BLOOMBERG_FILE = RAW_DATA_DIR / "bloomberg_data_session4_extended_2025_december.xlsx"
DVOL_FILE = RAW_DATA_DIR / "btc_dvol_complete_2021_2025.csv"
MACRO_FILE = RAW_DATA_DIR / "bloomberg_data_session7_final.xlsx"

# Sample definition
SAMPLE_START = "2022-01-01"

# Create generated-data/output directories if they do not exist
ANALYSIS_DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)