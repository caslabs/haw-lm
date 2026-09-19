import sys
from pathlib import Path
import mlflow
import yaml
import pandas as pd
from dotenv import load_dotenv


# This ensures script can find the parent directories for imports
sys.path.append(str(Path(__file__).resolve().parents[1]))

load_dotenv()

def get_best_run_params(experiment_name: str, model_name: str):
    """
    Return the params of the latest MLflow run for a given model name.
    """
    runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        filter_string=f"params.name = '{model_name}'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )

    if runs.empty:
        return None

    run = runs.iloc[0]
    params = {k.replace("params.", ""): v for k, v in run.items() if k.startswith("params.")}
    return params


def pretty_print_hyperparams(cfgs: list[dict], experiment_name: str = "mbert_train"):
    """
    Pretty-print hyperparameters for all models defined in the YAML config.
    """
    for cfg in cfgs:
        model_name = cfg["name"]
        params = get_best_run_params(experiment_name, model_name)
        print(f"\n📌 Model: {model_name}")
        print("-" * (8 + len(model_name)))
        if not params:
            print("⚠️  No params found.")
            continue

        for key, val in sorted(params.items()):
            print(f"{key:<25} : {val}")


def load_config(config_path: str = "multi_model_experiment.yaml") -> list[dict]:
    config_file = Path(__file__).resolve().parents[2] / config_path
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    with open(config_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["experiments"]


if __name__ == "__main__":
    cfgs = load_config()
    pretty_print_hyperparams(cfgs)
