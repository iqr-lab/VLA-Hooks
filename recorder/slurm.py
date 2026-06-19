from pathlib import Path
import subprocess

from recorder.config import load_yaml


def submit_sbatch(
    *,
    experiment_path: str,
    slurm_config_path: str = "configs/slurm.yaml",
):
    slurm = load_yaml(slurm_config_path)

    wrapper = f"""#!/bin/bash
#SBATCH --job-name=openpi-run
#SBATCH --partition={slurm['partition']}
#SBATCH --gpus={slurm['gpus']}
#SBATCH --cpus-per-task={slurm['cpus_per_task']}
#SBATCH --mem={slurm['mem']}
#SBATCH --time={slurm['time']}
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

source .venv/bin/activate

python scripts/run_experiment.py \
  --experiment {experiment_path}
"""

    Path("logs").mkdir(exist_ok=True)

    script_path = Path("logs") / "temp_sbatch.sh"
    script_path.write_text(wrapper)

    subprocess.run(
        ["sbatch", str(script_path)],
        check=True,
    )