from dotenv import load_dotenv
from pathlib import Path
import yaml
from .leaderboard import leaderboard  # relative import since same folder


def generate_leaderboard_from_config(config_path: str = "multi_model_experiment.yaml", push_discord: bool = True):
    """
    Generate and visualize a leaderboard of model performance based on previously logged MLflow experiments.

    This function:
    - Loads a YAML configuration file containing experiment definitions.
    - Queries MLflow for evaluation metrics of each model (e.g., perplexity, POS accuracy).
    - Generates and saves visualizations (bar charts and a metrics table).
    - Optionally sends the leaderboard results to a configured Discord webhook.

    Parameters:
    -----------
    config_path : str
        Path to the YAML config file containing the list of experiments.
    push_discord : bool
        Whether to send the leaderboard result to Discord (default: True).
    """

    # Load environment variables (for DISCORD_HOOK, etc.)
    load_dotenv()

    # Resolve config file path
    config_file = Path(config_path).resolve()
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    # Load experiment configurations
    with open(config_file, "r", encoding="utf-8") as f:
        cfgs = yaml.safe_load(f)["experiments"]

    # Generate leaderboard
    leaderboard(cfgs, push_discord=push_discord)


if __name__ == "__main__":
    generate_leaderboard_from_config()
