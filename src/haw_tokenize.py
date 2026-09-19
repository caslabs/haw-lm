from pathlib import Path
from typing import Optional
from transformers import AutoTokenizer
from datasets import load_dataset, DatasetDict


def run(
    raw_path: str,
    out_dir: str,
    model_name: str,
    max_length: int,
    tokenizer_path: Optional[str] = None,
) -> str:
    """
    Tokenise the cleaned corpus into a Hugging Face Arrow dataset.

    If `tokenizer_path` points to an on-disk tokenizer folder, that tokenizer is
    used. Otherwise the checkpoint tokenizer designated by `model_name` is used.
    The result is saved to `out_dir` (an Arrow dataset).

    Parameters
    ----------
    raw_path: str
        Path to plain-text corpus (one sentence per line).
    out_dir: str
        Target folder to store the Arrow dataset.
    model_name: str
        HF checkpoint name (e.g. "distilbert-base-multilingual-cased").
    max_length: int
        Truncation / padding length.
    tokenizer_path: str | None
        Local tokenizer folder to load (for scratch models).  If None or not
        found, the checkpoint tokenizer is used instead.

    Returns
    -------
    str
        `out_dir`, so Prefect can pass it downstream if needed.
    """

    # Decide which tokenizer to load
    if (
        tokenizer_path
        and isinstance(tokenizer_path, (str, Path))
        and Path(tokenizer_path).exists()
    ):
        load_id = str(tokenizer_path)
    else:
        load_id = model_name

    # Load tokenizer (fast if available, slow otherwise)
    tok = AutoTokenizer.from_pretrained(load_id, use_fast=True)

    # Load corpus as HF dataset
    ds = load_dataset("text", data_files=raw_path)  # {"train": Dataset}

    # Tokenise
    def tok_fn(batch):
        """
        Tokenize a batch of sentences.
        This function is applied to each batch of the dataset
        using the `map` method.
        """
        return tok(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    # Tokenize the dataset
    ds_tok: DatasetDict = ds.map(tok_fn, batched=True, remove_columns=["text"])

    # Save Arrow dataset
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    ds_tok.save_to_disk(str(out_path))

    return str(out_path)
