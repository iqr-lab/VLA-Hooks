# openpi-inference-recorder

This repo runs inference for OpenPI-based policies and records policy outputs / hook data.

No model code is in this repo. It lives in the submodules:

- `external/openpi-pi05-hooks`
- `external/openpi-pi0fast-hooks`

The runner can launch either model repo, start `serve_policy.py`, run LIBERO evaluation, and save recorded outputs to a configured record directory.

---

## Repo structure

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
├── external/
│   ├── openpi-pi05-hooks/
│   └── openpi-pi0fast-hooks/
├── recorder/
│   ├── config.py
│   ├── paths.py
│   ├── process.py
│   ├── server.py
│   ├── libero.py
│   ├── slurm.py
│   └── run.py
└── scripts/
    └── run_experiment.py
```

## 1. Clone Setup

After cloning the repo:

```bash
cd openpi-inference-recorder
```

Initialize submodules:

```bash
git submodule update --init --recursive
```

Create a python environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

This environment does not hold openpi or libero dependencies. They're expected in the `.sif` containers.

---

## 2. Configure containers

Edit:

```bash
configs/containers.yaml
```

Example:

```yaml
use_apptainer: true

server_sif: /absolute/path/to/openpi_server.sif
libero_sif: /absolute/path/to/libero.sif

scratch_root: /nfs/roberts/scratch

pythonpath: /app/src:/app/third_party/libero:/app/packages/openpi-client/src
```

### Options

| Field           | Meaning                                                                                                                           |
| --------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `use_apptainer` | Whether to run inside `.sif` containers. Usually `true` on the cluster.                                                           |
| `server_sif`    | SIF image used to run `scripts/serve_policy.py`.                                                                                  |
| `libero_sif`    | SIF image used to run LIBERO evaluation.                                                                                          |
| `scratch_root`  | Scratch filesystem mounted into the container. Record dirs should usually live here.                                              |
| `pythonpath`    | Python path used inside the container. Usually should stay as `/app/src:/app/third_party/libero:/app/packages/openpi-client/src`. |

---

## 3. Configure model repos

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

### Options

| Field                   | Meaning                                           |
| ----------------------- | ------------------------------------------------- |
| `repo`                  | Path to the model repo/submodule.                 |
| `serve_policy`          | Path to `serve_policy.py` inside that repo.       |
| `default_policy_config` | Default OpenPI policy config name for that model. |

---

## 4. Configure an experiment

Experiment configs live in:

```bash
configs/experiments/
```

Example `configs/experiments/pi05_libero.yaml`:

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

record_root: /nfs/roberts/scratch/pi_tkf6/as4643/policy_records
log_root: logs

use_apptainer: true
```

Example `configs/experiments/pi0fast_libero.yaml`:

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

record_root: /nfs/roberts/scratch/pi_tkf6/as4643/policy_records
log_root: logs

use_apptainer: true
```

### Options

| Field           | Meaning                                                                                       |
| --------------- | --------------------------------------------------------------------------------------------- |
| `name`          | Name used for output folders. A timestamp is appended automatically.                          |
| `model`         | Which model repo to use. Must match a key in `configs/models.yaml`, e.g. `pi05` or `pi0fast`. |
| `policy_config` | OpenPI policy config passed to `--policy.config`.                                             |
| `checkpoint`    | Checkpoint directory passed to `--policy.dir`.                                                |
| `env`           | Environment passed to `serve_policy.py`. Usually `LIBERO`.                                    |
| `task_suite`    | LIBERO task suite, e.g. `libero_10`, `libero_spatial`, `libero_object`, `libero_goal`.        |
| `num_trials`    | Number of trials per LIBERO task.                                                             |
| `port`          | Port used by the policy server.                                                               |
| `host`          | Host used by the LIBERO client. Usually `127.0.0.1`.                                          |
| `hook_config`   | Hook YAML mounted into the server container.                                                  |
| `record_root`   | Parent directory for recorded outputs. Use scratch, not project storage.                      |
| `log_root`      | Parent directory for logs.                                                                    |
| `use_apptainer` | Experiment-level override for container mode.                                                 |

The final output paths look like:

```text
record_root/name_YYYYMMDD_HHMMSS
log_root/name_YYYYMMDD_HHMMSS
```

For example:

```text
/nfs/roberts/scratch/pi_tkf6/as4643/policy_records/pi05_libero_20260618_203423
logs/pi05_libero_20260618_203423
```

---

## 5. Configure hooks

Hook configs live in:

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

The available hooks depend on what the model submodule supports.

---

## 6. Run interactively

Request an interactive GPU session:

```bash
salloc --partition=scavenge_gpu --gpus=1 --cpus-per-task=4 --mem=256G --time=4:00:00
```

Then run:

```bash
python scripts/run_experiment.py \
  --experiment configs/experiments/pi05_libero.yaml
```

or:

```bash
python scripts/run_experiment.py \
  --experiment configs/experiments/pi0fast_libero.yaml
```

---

## 7. Submit with sbatch

If Slurm submission support is enabled, configure:

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

Submit:

```bash
python scripts/run_experiment.py \
  --experiment configs/experiments/pi05_libero.yaml \
  --sbatch
```

or:

```bash
python scripts/run_experiment.py \
  --experiment configs/experiments/pi0fast_libero.yaml \
  --sbatch
```

The sbatch wrapper calls the same runner inside the job. It does not recursively submit another job.

---

## 8. CLI options

```bash
python scripts/run_experiment.py --help
```

Main options:

| Option         | Meaning                                                                      |
| -------------- | ---------------------------------------------------------------------------- |
| `--experiment` | Path to experiment YAML. Required.                                           |
| `--models`     | Path to model config YAML. Default: `configs/models.yaml`.                   |
| `--containers` | Path to container config YAML. Default: `configs/containers.yaml`.           |
| `--sbatch`     | Submit the run as a Slurm job instead of running immediately.                |
| `--slurm`      | Path to Slurm config YAML, if implemented. Default: `configs/slurm.yaml`.    |
| `--dry-run`    | Print the generated Slurm script/command without submitting, if implemented. |

---

## 9. What the runner does

For each experiment, the runner:

1. Loads experiment config.
2. Loads model config.
3. Loads container config.
4. Creates timestamped record/log directories.
5. Starts the selected model server:

   * `external/openpi-pi05-hooks/scripts/serve_policy.py`, or
   * `external/openpi-pi0fast-hooks/scripts/serve_policy.py`
6. Waits until the server is ready.
7. Runs LIBERO evaluation against the server.
8. Saves records to `record_dir`.
9. Saves logs to `log_dir`.
10. Stops the server.
