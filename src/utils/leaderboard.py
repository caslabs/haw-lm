import math
import os
from pathlib import Path
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from tabulate import tabulate
from discord_webhook import DiscordWebhook
import numpy as np


def leaderboard(cfgs,
                lm_experiment: str = "mbert_train",
                pos_experiment: str = "mbert_pos_finetune",
                perplexity_metric: str = "eval_loss",  # exp(loss) → ppl
                pos_metric: str = "eval_accuracy",
                pos_f1: str = "eval_f1",
                out_png: str = "artifacts/leaderboard.png",
                push_discord: bool | None = None):
    """Return a `pd.DataFrame` (and write a dual‑bar PNG).

    Parameters
    ----------
    cfgs : list[dict]
        The same list of experiment dicts you iterate over in *haw_flow.py*.
    lm_experiment / pos_experiment : str
        MLflow experiment names for LM pre‑training and POS fine‑tune.
    perplexity_metric : str
        Metric key logged during LM eval whose exp() gives perplexity.
    pos_metric : str
        Metric key logged during POS eval (e.g. accuracy or f1).
    out_png : str
        Where to save the bar‑chart.
    push_discord : bool | None
        If **True** always push to Discord, if **False** never, if **None**
        autodetect based on the `DISCORD_HOOK` env var.
    """

    rows: list[dict] = []
    for cfg in cfgs:
        mdl_name = cfg["name"]


        rows.append(dict(
            model = mdl_name,
            perplexity = _latest_metric(lm_experiment, mdl_name, perplexity_metric),
            pos_acc = _latest_metric(pos_experiment, mdl_name, pos_metric),
            eval_f1 =  _latest_metric(pos_experiment, mdl_name, "eval_f1"),
            psuedo_perpleixty = _latest_metric(lm_experiment, mdl_name, "pseudo_perplexity"),
            masked_acc = _latest_metric(lm_experiment, mdl_name, "eval_masked_accuracy"),
            top5 = _latest_metric(lm_experiment, mdl_name, "eval_masked_top5"),
            mrr = _latest_metric(lm_experiment, mdl_name, "eval_masked_mrr"),
        ))

    df = pd.DataFrame(rows).sort_values("pos_acc", ascending=False)
    _plot(df, out_png)

    table_png_path = "artifacts/leaderboard_table.png"
    save_leaderboard_table_as_png(df, table_png_path)

    # Optionally notify on Discord
    if push_discord is None:
        push_discord = bool(os.getenv("DISCORD_HOOK"))
    if push_discord and os.getenv("DISCORD_HOOK"):
        DiscordWebhook(
            os.getenv("DISCORD_HOOK"),
            content="🏆 **Leaderboard Updated**",
            files={
                Path(out_png).name: open(out_png, "rb"),
                Path(table_png_path).name: open(table_png_path, "rb")
            }
        ).execute()
    else:
        print("Discord notifications disabled (set DISCORD_HOOK env var to enable).")
    return df


def save_leaderboard_table_as_png(
        df: pd.DataFrame,
        out_png: str = "artifacts/leaderboard_table.png"):

    # Show metrics in a table with the following columns:
    show_cols = ["model",
                 "perplexity",
                 "pos_acc",
                 "eval_f1",
                 "psuedo_perpleixty",
                 "masked_acc",
                 "top5",
                 "mrr"]

    df = df[show_cols].copy()

    # Round numeric columns
    for col in show_cols[1:]:  # skip "model"
        df[col] = pd.to_numeric(df[col], errors="coerce").round(4)

    # Replace NaNs with dash ("–") for aesthetics
    df.fillna("–", inplace=True)

    # Build a table with the specified columns
    fig, ax = plt.subplots(
        figsize=(
            2.2 + 1.4 * len(show_cols),  # width scales with # of columns
            0.8 + 0.5 * len(df)          # height scales with # of rows
        )
    )
    ax.axis("off")

    tbl = ax.table(cellText=df.values,
                   colLabels=df.columns,
                   loc="center",
                   cellLoc="center",
                   colLoc="center")

    tbl.scale(1.1, 1.5)  # Widen the columns slightly
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)  # Reduce font size for better fit

    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")

    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150, bbox_inches="tight", transparent=True)
    plt.close(fig)


def _latest_metric(experiment_name: str, model_name: str, metric_key: str):
    """Return most‑recent scalar metric value (or *None*)."""
    runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        filter_string=f"params.name = '{model_name}'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if runs.empty:
        return None
    val = runs.iloc[0].get(f"metrics.{metric_key}")
    return float(val) if pd.notnull(val) else None


def _plot(df: pd.DataFrame, out_png: str):
    """Side-by-side horizontal bar chart (perplexity ↓ vs POS-acc ↑)."""

    # Convert columns to numeric, coercing errors to NaN
    for col in ["perplexity", "pos_acc"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # drop models that have neither metric logged yet
    df = df.dropna(subset=["perplexity", "pos_acc"], how="all")

    # Warn if no models have logged any metrics
    if df.empty:
        print("Leaderboard is empty – no runs logged yet.")
        return

    # Plotting the benchmarks
    fig, (ax1, ax2) = plt.subplots(ncols=2,
                                   figsize=(8, 3),
                                   constrained_layout=True)

    df.plot.barh(x="model", y="perplexity", ax=ax1, legend=False,
                 title="Perplexity ↓")
    df.plot.barh(x="model", y="pos_acc", ax=ax2, legend=False,
                 title="POS-Accuracy ↑", color="darkgreen")

    for ax in (ax1, ax2):
        ax.invert_yaxis()
        ax.set_xlabel("")

    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
