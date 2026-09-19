# ─── src/train_tokenizer.py ───────────────────────────────
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer, BertWordPieceTokenizer
from transformers import PreTrainedTokenizerFast

def run(
    corpus_file: str,
    out_dir: str,
    tokenizer_type: str = "byte_bpe",   # "byte_bpe" | "wordpiece"
    vocab_size: int = 32_000,
    min_frequency: int = 2,
    lowercase: bool = True,
    character_coverage: float = 1.0,    # only used for WordPiece
) -> str:
    """
    Train a tokenizer (Byte-BPE or WordPiece) and save it in `out_dir`.

    Returns
    -------
    str  –  Folder path that now contains vocab + tokenizer.json
    """
    out = Path(out_dir)
    if (out / "tokenizer.json").exists():
        print(f"✅ Reusing existing tokenizer at {out}")
        return str(out)

    out.mkdir(parents=True, exist_ok=True)

    if tokenizer_type == "byte_bpe":
        print("🔧 Training Byte-level BPE tokenizer …")
        tok = ByteLevelBPETokenizer()
        tok.train(
            files=[corpus_file],
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"],
        )

    elif tokenizer_type == "wordpiece":
        print("🔧 Training WordPiece tokenizer …")
        tok = BertWordPieceTokenizer(lowercase=lowercase)
        tok.train(
            files=[corpus_file],
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        )

    else:
        raise ValueError(f"tokenizer_type must be 'byte_bpe' or 'wordpiece', got {tokenizer_type}")

    # save vocab / merges (BPE) or vocab.txt (WP) and tokenizer.json
    tok.save_model(str(out))
    tok_json = out / "tokenizer.json"
    tok.save(str(tok_json))

    # wrap in HF fast tokenizer
    hf_tok = PreTrainedTokenizerFast(
        tokenizer_file=str(tok_json),
        unk_token="[UNK]", pad_token="[PAD]",
        cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]",
    )
    hf_tok.save_pretrained(out)
    print("🎉 Tokenizer training complete\n")

    return str(out)
