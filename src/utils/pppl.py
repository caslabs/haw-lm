import evaluate

def compute_pppl(model_dir: str, texts: list[str], batch_size: int = 16) -> float:
    """
    Pseudo-perplexity for any BERT-style MLM.

    Parameters
    ----------
    model_dir : str
        HF checkpoint folder or model hub id.
    texts : list[str]
        Sentences to score (must be non-empty).
    batch_size : int
        Sentences per forward-pass group (trade-off: GPU RAM vs speed).
    """
    if not texts:
        raise ValueError("PPPL needs at least one sentence.")

    metric = evaluate.load("perplexity", model_type="mlm")
    result = metric.compute(
        predictions       = texts,
        model_id          = model_dir,
        batch_size        = batch_size,
        add_start_token   = False          # REQUIRED for MLMs
    )
    return result["perplexity"]
