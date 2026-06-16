#!/usr/bin/env python3

import argparse

from recorder.run import run_experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        required=True,
        help="Path to experiment YAML",
    )
    parser.add_argument(
        "--models",
        default="configs/models.yaml",
        help="Path to models YAML",
    )
    parser.add_argument(
        "--containers",
        default="configs/containers.yaml",
        help="Path to containers YAML",
    )

    args = parser.parse_args()

    run_experiment(
        experiment_path=args.experiment,
        models_path=args.models,
        containers_path=args.containers,
    )


if __name__ == "__main__":
    main()