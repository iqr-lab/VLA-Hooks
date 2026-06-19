# openpi-inference-recorder

This repo runs inference for OpenPI-based policies and records policy outputs and hook data.

No model code lives in this repository. Model implementations are included as git submodules:

* `external/openpi-pi05-hooks`
* `external/openpi-pi0fast-hooks`

The runner can:

* Launch either model repo
* Start `serve_policy.py`
* Run LIBERO evaluation
* Record policy outputs
* Save hook outputs
* Store logs and rollout artifacts
* Optionally submit runs through Slurm

---

# Repository Structure

```text
openpi-inference-recorder/
├── configs/
│   ├── containers.yaml
│   ├── models.yaml
│   ├── slurm.yaml
│   ├── hooks/
│   │   └── default.yaml
│   └── experiments/
│       ├── pi05_libero.yaml
│       └── pi0fast_libero.yaml
│
├── external/
│   ├── openpi-pi05-hooks/
│   └── openpi-pi0fast-hooks/
│
├── recorder/
│   ├── config.py
│   ├── paths.py
│   ├── process.py
│   ├── server.py
│   ├── libero.py
│   ├── slurm.py
│   └── run.py
│
└── scripts/
    └── run_experiment.py
```

---

# 1. Clone Setup

Clone the repository:

```bash
git clone <repo_url>
cd openpi-inference-recorder
```

Initialize submodules:

```bash
git submodule update --init --recursive
```

Create a lightweight Python environment:

```bash
python -m venv .venv
source .venv/bin/activate

pip install -e .
```

This environment only contains orchestration code.

OpenPI, LIBERO, JAX, CUDA, and related dependencies are expected to be provided by the configured `.sif` containers.

---

# 2. Configure Containers

Edit:

```bash
configs/containers.yaml
```

Example:

```yaml
use_apptainer: true

server_sif: /absolute/path/to/openpi_server.sif
libero_sif: /absolute/path/to/libero.sif

pythonpath: /app/src:/app/third_party/libero:/app/packages/openpi-client/src

mounts:
  - /path/to/checkpoints
  - /path/to/recordings
```

## Container Configuration Options

| Field           | Meaning                                                                 |
| --------------- | ----------------------------------------------------------------------- |
| `use_apptainer` | Whether to run inside `.sif` containers. Usually `true` on a cluster.   |
| `server_sif`    | SIF image used to run `scripts/serve_policy.py`.                        |
| `libero_sif`    | SIF image used to run LIBERO evaluation.                                |
| `pythonpath`    | Python path used inside the container. Usually should remain unchanged. |
| `mounts`        | List of filesystem paths that should be visible inside the container.   |

## What are mounts?

Containers cannot automatically access files on the host machine.

The `mounts` field tells Apptainer which directories should be shared with the container.

Example:

```yaml
mounts:
  - /nfs/roberts/scratch
  - /nfs/roberts/project
```

This allows the container to access files stored under:

```text
/nfs/roberts/scratch
/nfs/roberts/project
```

Any path referenced by:

* checkpoints
* datasets
* recordings
* output directories

should be included in `mounts`.

For example:

```yaml
checkpoint: /data/checkpoints/pi05/29999

record_root: /data/policy_records
```

requires:

```yaml
mounts:
  - /data
```

---

# 3. Configure Model Repositories

Edit:

```bash
configs/models.yaml
```

Example:

```yaml
models:
  pi0fast:
    repo: external/openpi-pi0fast-hooks
    serve_policy: scripts/serve_policy.py
    default_policy_config: pi0_fast_libero

  pi05:
    repo: external/openpi-pi05-hooks
    serve_policy: scripts/serve_policy.py
    default_policy_config: pi05_libero
```

## Model Configuration Options

| Field                   | Meaning                                                |
| ----------------------- | ------------------------------------------------------ |
| `repo`                  | Path to the model submodule.                           |
| `serve_policy`          | Path to `serve_policy.py` inside the model repository. |
| `default_policy_config` | Default OpenPI policy configuration for the model.     |

---

# 4. Configure Experiments

Experiment configurations live in:

```bash
configs/experiments/
```

## Example: Pi0.5

```yaml
name: pi05_libero

model: pi05

policy_config: pi05_libero
checkpoint: /absolute/path/to/checkpoint

env: LIBERO
task_suite: libero_10
num_trials: 1

port: 21000
host: 127.0.0.1

hook_config: configs/hooks/default.yaml

record_root: /path/to/policy_records
log_root: logs

use_apptainer: true
```

## Example: Pi0 Fast

```yaml
name: pi0fast_libero

model: pi0fast

policy_config: pi0_fast_libero
checkpoint: /absolute/path/to/checkpoint

env: LIBERO
task_suite: libero_10
num_trials: 1

port: 21000
host: 127.0.0.1

hook_config: configs/hooks/default.yaml

record_root: /path/to/policy_records
log_root: logs

use_apptainer: true
```

## Experiment Configuration Options

| Field           | Meaning                                                                   |
| --------------- | ------------------------------------------------------------------------- |
| `name`          | Name used for output directories.                                         |
| `model`         | Model key from `configs/models.yaml`.                                     |
| `policy_config` | OpenPI policy configuration passed to `--policy.config`.                  |
| `checkpoint`    | Checkpoint directory passed to `--policy.dir`.                            |
| `env`           | Environment name passed to `serve_policy.py`.                             |
| `task_suite`    | LIBERO task suite.                                                        |
| `num_trials`    | Number of evaluation rollouts per task.                                   |
| `port`          | Policy server port.                                                       |
| `host`          | Policy server host.                                                       |
| `hook_config`   | Hook configuration file.                                                  |
| `record_root`   | Parent directory for recordings. Must be accessible inside the container. |
| `log_root`      | Parent directory for logs.                                                |
| `use_apptainer` | Experiment-level override for container mode.                             |

Supported task suites depend on the underlying LIBERO version but commonly include:

```text
libero_spatial
libero_object
libero_goal
libero_10
libero_90
```

---

# Output Directories

Timestamped directories are automatically created:

```text
record_root/name_YYYYMMDD_HHMMSS
log_root/name_YYYYMMDD_HHMMSS
```

Example:

```text
/path/to/policy_records/pi05_libero_20260618_203423
logs/pi05_libero_20260618_203423
```

---

# 5. Configure Hooks

Hook configurations live in:

```bash
configs/hooks/
```

Example:

```yaml
enabled_hooks:
  - observation_input
  - prefix_final_hidden_state
  - final_action_chunk
  - raw_attention_weights
  - token_spans

raw_attention_weights:
  layers: null
  heads: null

prefix_gradients:
  enabled: false
  num_samples: 8

recording:
  save_inputs: true
  save_outputs: true
  save_hooks: true
```

Available hooks depend on what is implemented in the selected model repository.

---

# 6. Run Interactively

Request an interactive GPU allocation:

```bash
salloc \
  --partition=scavenge_gpu \
  --gpus=1 \
  --cpus-per-task=4 \
  --mem=256G \
  --time=4:00:00
```

Run Pi0.5:

```bash
python scripts/run_experiment.py \
  --experiment configs/experiments/pi05_libero.yaml
```

Run Pi0 Fast:

```bash
python scripts/run_experiment.py \
  --experiment configs/experiments/pi0fast_libero.yaml
```

---

# 7. Submit Through Slurm

Configure:

```bash
configs/slurm.yaml
```

Example:

```yaml
partition: scavenge_gpu
gpus: 1
cpus_per_task: 4
mem: 256G
time: "4:00:00"
```

Submit a job:

```bash
python scripts/run_experiment.py \
  --experiment configs/experiments/pi05_libero.yaml \
  --sbatch
```

or

```bash
python scripts/run_experiment.py \
  --experiment configs/experiments/pi0fast_libero.yaml \
  --sbatch
```

The generated Slurm job invokes the same runner within the allocation.

---

# 8. CLI Options

View help:

```bash
python scripts/run_experiment.py --help
```

Main options:

| Option         | Meaning                                          |
| -------------- | ------------------------------------------------ |
| `--experiment` | Path to experiment YAML. Required.               |
| `--models`     | Path to model configuration YAML.                |
| `--containers` | Path to container configuration YAML.            |
| `--sbatch`     | Submit via Slurm instead of running immediately. |
| `--slurm`      | Path to Slurm configuration file.                |
| `--dry-run`    | Print generated Slurm script without submitting. |

---

# 9. Execution Flow

For each experiment the runner:

1. Loads experiment configuration.
2. Loads model configuration.
3. Loads container configuration.
4. Creates timestamped output directories.
5. Starts the selected policy server.
6. Waits for server readiness.
7. Runs LIBERO evaluation.
8. Saves recordings.
9. Saves logs.
10. Stops the server.

Policy servers are launched from:

```text
external/openpi-pi05-hooks/scripts/serve_policy.py
```

or

```text
external/openpi-pi0fast-hooks/scripts/serve_policy.py
```

depending on the selected model.

---

# 10. Updating Submodules

Pull latest model code:

```bash
git submodule update --remote --merge
```

Commit updated submodule references:

```bash
git add external/openpi-pi05-hooks
git add external/openpi-pi0fast-hooks

git commit -m "Update OpenPI submodules"
```
