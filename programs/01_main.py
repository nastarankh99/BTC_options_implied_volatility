import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

scripts = [
    ROOT / "programs" / "01_dataprep" / "01_prepare_analysis_data.py",
    ROOT / "programs" / "02_analysis" / "02_main_analysis.py",
    ROOT / "programs" / "03_appendix" / "03_appendix.py",
]

for script in scripts:
    print(f"Running: {script.name}")

    subprocess.run(
        [sys.executable, "-u", str(script)],
        cwd=ROOT,
        check=True,
    )

print("Replication completed successfully.")
print("All main-paper and appendix results have been generated.")