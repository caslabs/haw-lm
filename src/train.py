import random, pathlib, torch, yaml
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForMaskedLM,
                          DataCollatorForLanguageModeling,
                          Trainer, TrainingArguments,
                          TrainerCallback, BertConfig, BertForMaskedLM)
import mlflow, torch, pathlib, random
from src.train_word2vec import train_word2vec
from src.train_tfidf import train_tfidf
import numpy as np
from torch.nn.functional import log_softmax
from transformers.trainer_utils import EvalPrediction
from transformers import EarlyStoppingCallback
import torch
from transformers.trainer_utils import EvalPrediction
import torch._dynamo
torch._dynamo.config.suppress_errors = True

@torch.no_grad()
def compute_metrics_ultra_efficient(pred: EvalPrediction, max_samples: int = 10000):
    """
    Ultra memory-efficient version that samples a subset of predictions for evaluation.
    Use this if you're still running out of memory with the above version.
    """
    logits_np = pred.predictions
    labels_np = pred.label_ids

    # Find all valid (non -100) positions
    valid_mask = labels_np != -100
    valid_indices = np.where(valid_mask)

    # If no valid positions, return zero metrics
    if len(valid_indices[0]) == 0:
        return {"masked_accuracy": 0.0, "masked_top5": 0.0, "masked_mrr": 0.0}

    # Sample a subset if too many valid positions
    if len(valid_indices[0]) > max_samples:
        sample_indices = np.random.choice(len(valid_indices[0]), max_samples, replace=False)
        batch_indices = valid_indices[0][sample_indices]
        seq_indices = valid_indices[1][sample_indices]
    else:
        batch_indices = valid_indices[0]
        seq_indices = valid_indices[1]

    # Extract only the sampled logits and labels
    sampled_logits = torch.tensor(logits_np[batch_indices, seq_indices], dtype=torch.float32)
    sampled_labels = torch.tensor(labels_np[batch_indices, seq_indices], dtype=torch.long)

    # Compute metrics
    preds = sampled_logits.argmax(-1)
    acc = preds.eq(sampled_labels).float().mean().item()

    # Add top-5 accuracy and MRR metrics for MLM once researched. For now,

    top5 = sampled_logits.topk(5, dim=-1).indices
    top5_acc = (top5.eq(sampled_labels.unsqueeze(-1)).any(-1).float().mean().item())

    # MRR calculation
    mrr_sum = 0.0
    for i in range(len(sampled_logits)):
        gold_logit = sampled_logits[i, sampled_labels[i]]
        rank = (sampled_logits[i] > gold_logit).sum().item() + 1
        mrr_sum += 1.0 / rank

    mrr = mrr_sum / len(sampled_logits)

    return {
        "masked_accuracy": acc,
        "masked_top5": top5_acc,
        "masked_mrr": mrr,
    }

@torch.no_grad()
def calculate_pseudo_perplexity(model, tokenizer, validation_file, max_samples=200, device='cuda'):
    """
    Calculate pseudo perplexity for a masked language model using validation data.

    Parameters:
    - model: Trained masked language model
    - tokenizer: Corresponding tokenizer
    - validation_file: Path to validation text file
    - max_samples: Maximum number of sentences to evaluate
    - device: 'cuda' or 'cpu'

    Returns:
    - pseudo_perplexity: float
    """
    model.eval()
    model.to(device)

    # Read validation texts
    with open(validation_file, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]

    # Sample if too many texts
    if len(texts) > max_samples:
        texts = random.sample(texts, max_samples)

    total_log_prob = 0.0
    total_tokens = 0
    mask_token_id = tokenizer.mask_token_id

    # Skip special tokens
    special_tokens = {tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id}
    if hasattr(tokenizer, 'unk_token_id') and tokenizer.unk_token_id is not None:
        special_tokens.add(tokenizer.unk_token_id)
    for text in texts:
        # Tokenize the text
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        # Get sequence length
        seq_len = input_ids.size(1)

        # Skip very short sequences
        if seq_len <= 2:
            continue

        # Iterate over each token in the sequence
        for i in range(seq_len):
            token_id = input_ids[0, i].item()

            # Skip special tokens
            if token_id in special_tokens:
                continue

            # Create masked version
            masked_input = input_ids.clone()
            masked_input[0, i] = mask_token_id

            try:
                # Get predictions
                outputs = model(masked_input, attention_mask=attention_mask)
                logits = outputs.logits[0, i]  # Logits for the masked position

                # Get log probability of the original token
                log_probs = log_softmax(logits, dim=-1)
                token_log_prob = log_probs[token_id].item()

                total_log_prob += token_log_prob
                total_tokens += 1

            except Exception as e:
                print(f"Error processing token {token_id}: {e}")
                continue

    # Return infinity if no tokens processed
    if total_tokens == 0:
        return float('inf')

    # Calculate pseudo perplexity
    avg_log_prob = total_log_prob / total_tokens
    pseudo_ppl = torch.exp(torch.tensor(-avg_log_prob)).item()

    return pseudo_ppl


class MLflowCallback(TrainerCallback):
    """
    A simple callback to log metrics to MLflow during training.
    """
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        for k, v in logs.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, float(v), step=state.global_step)


def _prepare_files(corpus_path: str, train_split: float, seed: int):
    """
    Prepare train/valid text files from a corpus file.
    """
    lines = [l.strip() for l in open(corpus_path, encoding="utf-8") if l.strip()]
    random.seed(seed)
    random.shuffle(lines)

    split_idx = int(len(lines) * train_split)
    train_lines, val_lines = lines[:split_idx], lines[split_idx:]

    pathlib.Path("quick").mkdir(exist_ok=True)
    pathlib.Path("quick/train.txt").write_text("\n".join(train_lines), encoding="utf-8")
    pathlib.Path("quick/valid.txt").write_text("\n".join(val_lines), encoding="utf-8")
    return "quick/train.txt", "quick/valid.txt"


def build_tokenizer(cfg):
    """
    Priority:
      1) cfg['tokenizer_path']  – use custom tokenizer path
      2) cfg['model_name']  – use pre-trained tokenizer from HuggingFace
    """
    if cfg.get("tokenizer_path"):
        return AutoTokenizer.from_pretrained(cfg["tokenizer_path"])
    return AutoTokenizer.from_pretrained(cfg["model_name"])


from transformers import (
    BertConfig,
    BertForMaskedLM,
    AutoModelForMaskedLM,
)

def build_model(cfg: dict):
    """
    Build a BERT-based masked-language model from `cfg`.

    • If `train_from_scratch` is True ➜ initialise Mini-BERT with the
      hyper-parameters supplied in the YAML.
    • Otherwise ➜ load a Hugging Face checkpoint in `cfg["model_name"]`.

    Required `cfg` keys for scratch builds
    ──────────────────────────────────────
      vocab_size               – size of WordPiece/SentencePiece vocab
      hidden_size              – Transformer hidden dimension
      num_hidden_layers        – depth of the encoder
      num_attention_heads      – heads per layer  (hidden_size ÷ heads must be an int)
      intermediate_size        – feed-forward dimension (≈ 4× hidden_size)
      max_position_embeddings  – positional-embedding length (e.g. 128)
    """
    if cfg.get("train_from_scratch", False):
        if cfg.get("model_type") == "bert":
            bert_cfg = BertConfig(
                vocab_size=cfg["vocab_size"],
                hidden_size=cfg["hidden_size"],
                num_hidden_layers=cfg["num_hidden_layers"],
                num_attention_heads=cfg["num_attention_heads"],
                intermediate_size=cfg["intermediate_size"],
                max_position_embeddings=cfg.get("max_position_embeddings",
                                                cfg.get("max_length", 512)),
                type_vocab_size=2,              # keep BERT’s segment IDs
                hidden_dropout_prob=0.10,
                attention_probs_dropout_prob=0.10,
                pad_token_id=0,
            )
            return BertForMaskedLM(bert_cfg)
        else:
            raise ValueError(f"Unsupported model_type: {cfg['model_type']}")
    else:
        return AutoModelForMaskedLM.from_pretrained(cfg["model_name"])


def hpo_search_space(trial):
    """
    Define the hyperparameter search space for Optuna.
    This function is called by Optuna to suggest hyperparameters for the training.
    1. learning_rate: Log-uniform distribution between 1e-5 and 5e-4
    2. per_device_train_batch_size: Categorical distribution of [8, 16, 32]
    3. weight_decay: Uniform distribution between 0.0 and 0.3
    4. warmup_ratio: Uniform distribution between 0.0 and 0.3
    5. mlm_probability: Uniform distribution between 0.1 and 0.3
    """
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8, 16, 32]),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
        "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.1),
        "mlm_probability": trial.suggest_float("mlm_probability", 0.1, 0.3),
    }


def run(tokenized_dir: str, cfg: dict):
    cfg.setdefault("checkpoint_path",
                   f"model_out/{cfg.get('name','model')}/pretrain")
    cfg.setdefault("pos_output_dir",
                   f"model_out/{cfg.get('name','model')}/pos")

    """
    Main training function for the masked language model.
    """
    if cfg.get("model_type") == "word2vec":
        # Word2Vec doesn’t need tokenization or MLM
        corpus_file = "data/clean.txt"
        if cfg.get("data_size"):
            # respect sampling knob
            corpus_file = "quick/subset.txt"

        metrics, model_dir = train_word2vec(corpus_file, cfg)
        return metrics, model_dir, None # no valid_txt for Word2Vec

    # TODO: Add support for other model types if needed
    if cfg.get("model_type") == "tfidf":
        corpus = "data/clean.txt"
        if cfg.get("data_size"):
            corpus = "quick/subset.txt"

        metrics, model_dir = train_tfidf(corpus, cfg)
        return metrics, model_dir, None

    # Load configuration from YAML
    for k in ["learning_rate","batch_size","max_steps","max_length",
            "train_split","mlm_probability","seed"]:
        if k not in cfg or cfg[k] is None:          # ← skip missing / null
            continue
        if k in ("train_split", "mlm_probability"):
            cfg[k] = float(cfg[k])
        elif k == "learning_rate":
            cfg[k] = float(cfg[k])
        else:
            cfg[k] = int(cfg[k])


   # Optional Subsample
    corpus_file = "data/clean.txt"
    if cfg.get("data_size"):
        tmp = pathlib.Path("quick/subset.txt")
        lines = [l.strip() for l in open(corpus_file, encoding="utf-8") if l.strip()]
        keep  = int(len(lines) * float(cfg["data_size"]))
        tmp.write_text("\n".join(lines[:keep]), encoding="utf-8")
        corpus_file = str(tmp)

    # Prepare train/valid files
    train_txt, valid_txt = _prepare_files(
        corpus_file, cfg["train_split"], cfg["seed"]
    )

    # Prepare tokenizer and datasets
    tok = build_tokenizer(cfg)
    ds  = load_dataset("text",
                       data_files={"train": train_txt, "validation": valid_txt})

    def tok_fn(b):
        """
        Tokenization function for the dataset.
        - Truncates to max_length
        - Returns tokenized input IDs
        - Uses the tokenizer's truncation and padding capabilities
        """
        return tok(b["text"], truncation=True, max_length=cfg["max_length"])

    # Tokenize the dataset
    token_ds = ds.map(tok_fn, batched=True, remove_columns=["text"])

    # Data collator & model
    collator = DataCollatorForLanguageModeling(
        tokenizer=tok, mlm=True, mlm_probability=cfg["mlm_probability"])
    model    = build_model(cfg)

    out_dir = pathlib.Path(cfg["checkpoint_path"]).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Training arguments
    args = TrainingArguments(
        output_dir=str(out_dir),
        overwrite_output_dir=True,
        #max_steps=cfg["max_steps"],
        per_device_train_batch_size=cfg["batch_size"],
        eval_strategy="steps",
        eval_steps=1000,
        # save_steps=1000,
        per_device_eval_batch_size=max(1, cfg["batch_size"] // 4),
        eval_accumulation_steps=1,
        prediction_loss_only=True,
        learning_rate=cfg["learning_rate"],
        logging_strategy="steps",
        logging_steps=50,
        save_strategy="no",
        # save_total_limit=1,
        fp16=torch.cuda.is_available(),
        seed=cfg["seed"],
        report_to="none",
        load_best_model_at_end=False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )


    def model_init():
        """
        Model initialization function for HPO.
        This is required for Optuna hyperparameter search to create a new model instance
        for each trial with the suggested hyperparameters.
        """
        return build_model(cfg)


    # HuggingFace Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=token_ds["train"],
        eval_dataset=token_ds["validation"],
        data_collator=collator,
        tokenizer=tok,
        compute_metrics=compute_metrics_ultra_efficient,
        callbacks=[
            MLflowCallback(),
            EarlyStoppingCallback(early_stopping_patience=2),
        ],
    )

    # Hyperparameter Optimization (HPO) with Optuna
    # ── Hyper-parameter search (Optuna) ──────────────────────────────
    if cfg.get("search", False):
        try:
            best_run = Trainer(
                args=args,
                tokenizer=tok,
                train_dataset=token_ds["train"],
                eval_dataset=token_ds["validation"],
                data_collator=collator,
                compute_metrics=compute_metrics_ultra_efficient,
                callbacks=[
                    MLflowCallback(),
                    EarlyStoppingCallback(early_stopping_patience=2),
                ],
                model_init=model_init,
            ).hyperparameter_search(
                backend="optuna",
                n_trials=10,
                direction="minimize",
                hp_space=hpo_search_space,
                compute_objective=lambda m: m["eval_loss"],
            )
        except ValueError:
            print("⚠️  All Optuna trials failed – falling back to base hyper-params")
            best_run = None

        if best_run is not None:
            print(f"Best HPO run: {best_run.hyperparameters}")
            # transfer the suggested params onto both cfg *and* trainer.args
            for k, v in best_run.hyperparameters.items():
                setattr(trainer.args, k, v)
                cfg[k] = v

            # Re-create the MLM collator if mlm_probability changed
            collator = DataCollatorForLanguageModeling(
                tokenizer=tok,
                mlm=True,
                mlm_probability=cfg["mlm_probability"],
            )

            # Re-instantiate trainer so the new args are really used
            trainer = Trainer(
                model=model,
                args=trainer.args,
                train_dataset=token_ds["train"],
                eval_dataset=token_ds["validation"],
                data_collator=collator,
                tokenizer=tok,
                compute_metrics=compute_metrics_ultra_efficient,
                callbacks=[
                    MLflowCallback(),
                    EarlyStoppingCallback(early_stopping_patience=2),
                ],
            )


    trainer.train()
    loss_only_metrics = trainer.evaluate()

    # We select a subset of validation set for evaluation to avoid Out of Memory issues
    eval_subset_size = min(1000, len(token_ds["validation"]))
    small_eval_set = token_ds["validation"].select(range(eval_subset_size))

    full_eval_args = TrainingArguments(
        output_dir               = str(out_dir),
        per_device_eval_batch_size = max(1, cfg["batch_size"] // 2),
        prediction_loss_only     = False,
        eval_accumulation_steps  = 4,
        dataloader_drop_last     = False,
        fp16                     = torch.cuda.is_available(),
        report_to                = "none",
        save_strategy            = "no",
    )


    # Full evaluation with metrics
    # This is done on a smaller subset to avoid OOM issues
    full_eval_trainer = Trainer(
        model         = model,
        args          = full_eval_args,
        eval_dataset  = small_eval_set,
        data_collator = collator,
        compute_metrics = compute_metrics_ultra_efficient,
    )
    full_metrics = full_eval_trainer.evaluate()
    metrics = {**loss_only_metrics, **full_metrics}

    # TODO: Research these metrics further
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        pseudo_ppl = calculate_pseudo_perplexity(model, tok, valid_txt, max_samples=100, device=device)  # ✅ reduced sample size
        traditional_ppl = torch.exp(torch.tensor(metrics["eval_loss"])).item()

        metrics["pseudo_perplexity"] = pseudo_ppl
        metrics["traditional_perplexity"] = traditional_ppl

        print(f"Traditional Perplexity: {traditional_ppl:.2f}")
        print(f"Pseudo Perplexity: {pseudo_ppl:.2f}")

    except Exception as e:
        print(f"Error calculating pseudo perplexity: {e}")
        metrics["traditional_perplexity"] = torch.exp(torch.tensor(metrics["eval_loss"])).item()

    # Optionally sanitize metrics before logging
    safe_metrics = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}

    trainer.save_model(out_dir)
    tok.save_pretrained(out_dir)

    return safe_metrics, str(out_dir), valid_txt
