import yaml, os, mlflow
from prefect import flow, task, get_run_logger
from discord_webhook import DiscordWebhook
from pathlib import Path
from dotenv import load_dotenv
import traceback
from src import clean, haw_tokenize, train, evaluate, mask_interact, pos_finetune, pos_interact, train_tokenizer
from src.utils.leaderboard import leaderboard
from src.utils.flatten import flatten_metrics
import random
import argparse
load_dotenv()

def sample_corpus(corpus_file, output_file, fraction=1.0, shuffle=True, seed=42):
    """
    Sample a given fraction of lines from a corpus file and write them to an output file.
    If fraction is 1.0, the entire file is used (shuffled or not).

    Parameters:
    - corpus_file: Path to input file
    - output_file: Path to write sampled output
    - fraction: Float between 0 and 1 (1.0 = use all)
    - shuffle: Whether to shuffle the lines before sampling
    - seed: Random seed (for reproducibility if shuffle is True)
    """
    with open(corpus_file, 'r', encoding='utf-8') as f:
        lines = [line for line in f if line.strip()]  # Skip blank lines

    if shuffle:
        random.seed(seed)
        random.shuffle(lines)

    num_lines_to_sample = int(len(lines) * fraction)
    sampled_lines = lines[:num_lines_to_sample]

    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(sampled_lines)

    print(f"📦 Sampled {len(sampled_lines)} lines from `{corpus_file}` → `{output_file}` (fraction={fraction}, shuffle={shuffle})")


@task(retries=2, log_prints=True)
def step_clean(cfg):
    """
    Clean the raw Hawaiian no text.
    """
    clean.run("data/corpus_1x.txt", "data/clean.txt", cfg)


@task
def step_tokenize(cfg):
    """
    Tokenize the cleaned text.
    """
    haw_tokenize.run(
        "data/clean.txt",
        "data/tok",
        cfg["model_name"],
        cfg["max_length"],
        tokenizer_path=cfg.get("tokenizer_path"),
    )


@task
def step_train(cfg):
    """
    Train the model on the tokenized dataset.
    This will log parameters and metrics to MLflow.
    """
    mlflow.set_experiment("mbert_train")
    with mlflow.start_run() as run:
        mlflow.log_params(cfg)
        metrics, model_dir, valid_txt = train.run("data/tok", cfg)
        mlflow.log_metrics(flatten_metrics(metrics))
        return run.info.run_id, metrics, model_dir, valid_txt


@task
def step_evaluate(metrics):
    """
    Evaluate the model on the validation set.
    This will return the perplexity and other metrics.
    """
    return evaluate.run(metrics)

@task
def step_interact(model_dir: str, repl_enabled: bool):
    if repl_enabled:
        mask_interact.run(model_dir)

@task
def notify_experiment_start(cfg: dict, discord_enabled: bool):
    """
    Notify that an experiment is starting.
    This will send a message to Discord if enabled.
    """
    if not discord_enabled:
        return

    hook = os.getenv("DISCORD_HOOK")
    if not hook:
        print("⚠️ Discord webhook not configured. Skipping experiment start notification.")
        return

    msg = [
        "**Experiment Started**",
        f"Model: `{cfg.get('model_name')}`",
        f"Seed: `{cfg.get('seed')}`",
        f"Checkpoint Path: `{cfg.get('checkpoint_path')}`",
        f"POS Output Dir: `{cfg.get('pos_output_dir')}`",
        "",
    ]
    DiscordWebhook(hook, content="\n".join(msg)).execute()


@task
def notify_training_start(cfg: dict, discord_enabled: bool):
    """
    Notify that training is starting.
    This will send a message to Discord if enabled.
    """
    if not discord_enabled:
        return

    hook = os.getenv("DISCORD_HOOK")
    if not hook:
        print("⚠️ Discord webhook not configured. Skipping training start notification.")
        return

    msg = [
        "",
        f"**Training `{cfg.get('model_name')}` on `{cfg.get('corpus_data_name')}**",
        "Corpus: `data/corpus.txt` (Kahupule dataset)",
        "Config:",
        f"• Max Steps: `{cfg.get('max_steps')}` | LR: `{cfg.get('learning_rate')}`",
        f"• MLM Prob: `{cfg.get('mlm_probability')}` | Max Length: `{cfg.get('max_length')}`",
        f"• Train Split: `{cfg.get('train_split')}`",
        f"Checkpoint: `{cfg.get('checkpoint_path')}`"
    ]
    DiscordWebhook(hook, content="\n".join(msg)).execute()


@task
def notify_finetune_start(cfg: dict, discord_enabled: bool):
    """
    Notify that fine-tuning for POS tagging is starting.
    This will send a message to Discord if enabled.
    """
    if not discord_enabled:
        return

    hook = os.getenv("DISCORD_HOOK")
    if not hook:
        print("⚠️ Discord webhook not configured. Skipping fine-tuning start notification.")
        return

    msg = [
        "",
        f"**Fine-tuning `{cfg.get('model_name')}-hawaiian-aware` for `{cfg.get('pos_data_name')}`**",
        f"POS Output Dir: `{cfg.get('pos_output_dir')}`",
        f"Dataset: `POS corpus` (.conllu, low-resource)"
    ]
    DiscordWebhook(hook, content="\n".join(msg)).execute()


@task
def notify_preprocessing_done(cfg: dict, discord_enabled: bool):
    """
    Notify that preprocessing is complete.
    This will send a message to Discord if enabled.
    """
    if not discord_enabled:
        return

    hook = os.getenv("DISCORD_HOOK")
    if not hook:
        print("⚠️ Discord webhook not configured. Skipping preprocessing notification.")
        return

    msg = [
        "",
        "**Preprocessing Complete**",
        f"Input Corpus: `{cfg.get('corpus_file', 'data/corpus.txt')}`",
        f"Tokenizer: `{cfg.get('model_name')}`",
        "Settings:",
        f"• Max Length: `{cfg.get('max_length')}`",
        f"• Train Split: `{cfg.get('train_split')}`",
        f"• Seed: `{cfg.get('seed')}`",
        "",
    ]
    corpus_data_clean = cfg.get('corpus_data_clean')
    if corpus_data_clean:
        msg.append("• Preprocessing Hypothesis:")
        if isinstance(corpus_data_clean, list):
            msg.extend([f"`{line}`" for line in corpus_data_clean])
        else:
            msg.append(f"`{corpus_data_clean}`")
        msg.append("")
    DiscordWebhook(hook, content="\n".join(msg)).execute()


@task
def notify(run_id, ppl, discord_enabled: bool):
    """
    Notify about the completion of a Prefect run.
    This will send a message to Discord if enabled.
    """
    if not discord_enabled:
        return

    hook = os.getenv("DISCORD_HOOK")
    if not hook:
        print("⚠️ Discord webhook not configured. Skipping notification.")
        return

    DiscordWebhook(hook,
        content=f"Prefect run `{run_id}` finished.\nPerplexity: `{ppl:.3f}`").execute()


@task
def step_pos_finetune(cfg):
    """
    Fine-tune the model for POS tagging.
    """
    mlflow.set_experiment("mbert_pos_finetune")

    with mlflow.start_run() as run:
        mlflow.log_param("lr", 0.001)
        mlflow.log_params(cfg)
        metrics = pos_finetune.run(cfg)
        mlflow.log_metrics(flatten_metrics(metrics))
        return run.info.run_id, metrics


@task
def step_pos_interact(cfg):
    """
    Launch the POS tagger REPL for interactive testing.
    """
    print("\n⚠️ [OPTIONAL] Launching POS tagger REPL")
    pos_interact.run(cfg)


@task
def notify_completion(run_id: str, metrics: dict, discord_enabled: bool, model_name: str = "Unknown"):
    """
    Notify about the completion of a training run.
    This will send a message to Discord if enabled.
    """
    if not discord_enabled:
        return

    hook = os.getenv("DISCORD_HOOK")
    if not hook:
        print("⚠️ Discord webhook not configured. Skipping completion notification.")
        return

    lines = [
        f"✅ Completed training `{model_name}`",
        f"🧪 Run ID: `{run_id}`"
    ]

    # Dynamically include universal metrics
    if metrics:
        lines.append("📊 Metrics:")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                lines.append(f"• {key}: `{value:.4f}`")
        lines.append("")

    DiscordWebhook(hook, content="\n".join(lines)).execute()


@task
def notify_error(error: Exception, discord_enabled: bool):
    """
    Notify about an error that occurred during the flow execution.
    """
    if not discord_enabled:
        return

    hook = os.getenv("DISCORD_HOOK")
    if not hook:
        print("⚠️ Discord webhook not configured. Skipping error notification.")
        return

    tb = "".join(traceback.format_exception(type(error), error, error.__traceback__))
    tb_short = tb.strip().split("\n")[-5:]  # show only last few lines

    msg = [
        "❌ **Flow Error Notification**",
        f"Exception: `{str(error)}`",
        "```" + "\n".join(tb_short) + "```"
    ]

    DiscordWebhook(hook, content="\n".join(msg)).execute()


@task
def step_build_tokenizer(cfg: dict) -> str | None:
    """
    Build (or reuse) a tokenizer and return its folder path.
    """
    # 1) explicit path → reuse
    path = cfg.get("tokenizer_path")
    if path and Path(path).exists():
        return path

    # 2) non-scratch models just load a HF tokenizer later
    if not cfg.get("train_from_scratch", False):
        return None

    # 3) scratch run → train chosen tokenizer
    out_dir = path or f"tokenizers/{cfg['name']}"
    return train_tokenizer.run(
        corpus_file="data/clean.txt",
        out_dir=out_dir,
        tokenizer_type=cfg.get("tokenizer_type", "byte_bpe"),
        vocab_size=cfg.get("vocab_size", 32_000),
        min_frequency=cfg.get("min_freq", 2),
        lowercase=cfg.get("lowercase", True),
        character_coverage=cfg.get("character_coverage", 1.0),
    )


@task
def step_leaderboard(cfgs: list[dict], discord_enabled: bool = False,  **lb_kwargs):
    """
    Generate and return the leaderboard for all configurations.
    This will also push to Discord if enabled.
    The `lb_kwargs` can include additional parameters like `pos_metric`.
    """

    return leaderboard(cfgs, push_discord=discord_enabled, **lb_kwargs)



@flow(name="Multi-Model Hawaiian NLP")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="multi_model_experiment.yaml", help="Path to experiment config file")
    args = parser.parse_args()

    all_cfgs = yaml.safe_load(open(args.config))["experiments"]

    for cfg in all_cfgs:
        feature_flags = cfg.get("feature_flags", {})
        discord_enabled = feature_flags.get("notifications", {}).get("discord_enabled", False)
        repl_enabled    = feature_flags.get("devtools", {}).get("repl_enabled", True)


        # Set the sampling fraction based on the configuration name
        sample_fraction = cfg['data_size']

        # Sample the corpus and use it for preprocessing
        sampled_corpus_file = f"data/corpus_{sample_fraction}x.txt"
        sample_corpus("data/raw_sentences_test.txt", sampled_corpus_file, fraction=sample_fraction, shuffle=False)

        try:
            # Build or reuse tokenizer, then inject path
            cfg = cfg.copy()
            cfg["tokenizer_path"] = step_build_tokenizer(cfg)

            # Preprocessing: clean and tokenize
            notify_experiment_start(cfg, discord_enabled)
            step_clean(cfg)
            if cfg.get("model_type") not in ("word2vec", "tfidf"):
                step_tokenize(cfg)

            notify_preprocessing_done(cfg, discord_enabled)

            # Training for Hawaiian-Awareness
            notify_training_start(cfg, discord_enabled)
            run_id, train_metrics, model_dir, valid_txt = step_train(cfg)

            cfg["tokenizer_path"] = model_dir
            static_models = ("word2vec", "tfidf")

            if cfg.get("model_type") not in static_models:
                ppl = step_evaluate(train_metrics)
            else:
                ppl = None

            notify_completion(run_id, train_metrics, discord_enabled, cfg.get("model_name"))

            if repl_enabled:
                step_interact(model_dir, repl_enabled)

            # Fine-tuning for POS tagging
            if cfg.get("model_type") == "no-pos":
                # Skip POS fine-tuning
                pos_run_id, pos_metrics = None, {}
                print(f"Skipping POS fine-tuning for {cfg.get('model_type')}")
            else:
                notify_finetune_start(cfg, discord_enabled)
                pos_run_id, pos_metrics = step_pos_finetune(cfg)
                notify_completion(pos_run_id, pos_metrics, discord_enabled, cfg.get("model_name"))

                if repl_enabled:
                    step_pos_interact(cfg)
                else:
                    pos_metrics = {}


        except Exception as e:
            notify_error(e, discord_enabled)
            raise

    # Final leaderboard
    step_leaderboard(all_cfgs, True, pos_metric="eval_accuracy", pos_f1="eval_f1")


if __name__ == "__main__":
    main()
