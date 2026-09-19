# haw-lm/label_studio_pos_backend/ls_csv_to_conllu.py
"""
CSV → CoNLL-U converter used by the active-learning loop.
Call `csv_to_conllu("in.csv", "out.conllu")`.
"""
import csv, json, re
from pathlib import Path
import unicodedata

# Use the same tokenization regex as your model
TOKEN_REGEX = re.compile(r"\w+ʻ?\w*|['ʻ\w]+|[^\w\s]", re.UNICODE)

def is_punctuation(text: str) -> bool:
    """Check if text is punctuation using Unicode categories - same as model"""
    if not text:
        return False
    for char in text:
        category = unicodedata.category(char)
        if not (category.startswith('P') or category.startswith('S')):
            return False
    return True

def _load_labels(label_json: str):
    if not isinstance(label_json, str) or not label_json.strip():
        return {}
    spans = json.loads(label_json)
    return {(s["start"], s["end"]): s["labels"][0]
            for s in spans if s.get("labels")}

def _find_best_label_match(token_start: int, token_end: int, token_text: str, labels: dict) -> str:
    """
    Find the best matching label for a token, handling tokenization mismatches.
    """
    # First try exact match
    exact_match = labels.get((token_start, token_end))
    if exact_match:
        return exact_match

    # Try to find overlapping spans
    for (label_start, label_end), label in labels.items():
        # Check if token is contained within a labeled span
        if label_start <= token_start and token_end <= label_end:
            return label

        # Check if token overlaps with labeled span
        if not (token_end <= label_start or token_start >= label_end):
            # There's overlap, use this label
            return label

    # Fallback: infer from token characteristics
    if is_punctuation(token_text):
        return "PUNCT"

    # If no match found, return a more appropriate default than "_"
    print(f"[csv_to_conllu] WARNING: No label found for token '{token_text}' at ({token_start}, {token_end})")
    return "X"  # Unknown/other - better than "_"

def csv_to_conllu(csv_in: str | Path, conllu_out: str | Path) -> None:
    csv_in, conllu_out = Path(csv_in), Path(conllu_out)

    # Read entire CSV into memory first
    with csv_in.open(newline="", encoding="utf-8") as f_in:
        reader = csv.DictReader(f_in)
        rows = list(reader)

    with conllu_out.open("w", encoding="utf-8") as f_out:
        for sent_id, row in enumerate(rows, start=1):
            text = row["text"]
            labels = _load_labels(row["label"])
            if not text.strip():
                continue

            print(f"[csv_to_conllu] Processing sentence {sent_id}: {text}")
            print(f"[csv_to_conllu] Available labels: {labels}")

            f_out.write(f"# sent_id = {sent_id}\n# text = {text}\n")

            for idx, match in enumerate(TOKEN_REGEX.finditer(text), 1):
                start, end = match.span()
                form = match.group()

                # Use improved label matching
                upos = _find_best_label_match(start, end, form, labels)

                print(f"[csv_to_conllu] Token '{form}' at ({start}, {end}) -> {upos}")

                cols = [idx, form, form.lower(), upos, "_", "_", 0, "dep", "_", "_"]
                f_out.write("\t".join(map(str, cols)) + "\n")
            f_out.write("\n")
