import subprocess
import sys
from pathlib import Path


# ECON 899 Replication Package Setup
# Bitcoin Option Expirations and Implied Volatility
# Run this file once on a new system before running the replication package.


ROOT = Path(__file__).resolve().parent.parent
REQUIREMENTS_FILE = ROOT / "requirements.txt"

# Create required directories if they do not already exist
required_directories = [
    ROOT / "data" / "raw_data",
    ROOT / "data" / "data_for_analysis",
    ROOT / "results",
]

for directory in required_directories:
    directory.mkdir(parents=True, exist_ok=True)

# Install the Python packages used in the analysis
print("Installing required Python packages...")
subprocess.check_call([
    sys.executable,
    "-m",
    "pip",
    "install",
    "-r",
    str(REQUIREMENTS_FILE),
])

print("\nSetup completed successfully.")
