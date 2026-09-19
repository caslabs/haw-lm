import os, glob, pathlib, sys
from transformers import pipeline, AutoTokenizer, AutoModelForMaskedLM

def _find_latest_ckpt(parent: str, pattern: str = "checkpoint-*") -> str | None:
    """
    Return the newest matching checkpoint inside `parent`, or None.
    """
    ckpts = sorted(glob.glob(os.path.join(parent, pattern)), key=os.path.getmtime)
    return ckpts[-1] if ckpts else None


def run(model_dir: str | None = None, top_k: int = 5):
    """
    Launch an interactive [MASK]-filling shell.

    Parameters
    ----------
    model_dir : str | None
        • If provided, that folder *must* contain a Hugging Face model
          (config.json, weights, tokenizer files, …).
        • If None, we try:
            1. $HAW_LM_CHECKPOINT  environment variable
            2. newest checkpoint inside ./quick-haw-mlm/
            3. raise a helpful error
    top_k : int
        How many predictions to show.
    """


    # Retrieve model directory
    if model_dir is None:
        model_dir = (
            os.getenv("HAW_LM_CHECKPOINT")
            or _find_latest_ckpt("quick-haw-mlm")
        )

    if model_dir is None or not pathlib.Path(model_dir, "config.json").exists():
        msg = (
            "❌  Cannot locate a valid checkpoint to load.\n"
            "    • Pass the path in run(model_dir='...')\n"
            "    • or set $HAW_LM_CHECKPOINT\n"
            "    • or ensure quick-haw-mlm/checkpoint-* exists"
        )
        sys.exit(msg)

    # Build the fill-mask pipeline
    fm = pipeline(
        "fill-mask",
        model     = AutoModelForMaskedLM.from_pretrained(model_dir),
        tokenizer = AutoTokenizer.from_pretrained(model_dir),
        top_k     = top_k,
        device_map= "auto",
    )

    # REPL
    print("Hawaiian MLM • type a sentence containing [MASK] (or /exit to quit)")
    while True:
        try:
            sent = input("» ")
            if sent.strip().lower() in {"/exit", ""}:
                print("Exiting…"); break
            for out in fm(sent):
                print(f" {out['token_str']:15} (p={out['score']:.3f})")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting…"); break
        except Exception as e:
            print(f"Error: {e}")
