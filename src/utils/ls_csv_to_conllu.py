"""
ls_csv_to_conllu.py
Convert a Label Studio span-level CSV export (text + JSON 'label' column)
into CoNLL-U.

Assumptions based on how 2025 Label Studio works with the POS model:
-----------
* You used the NER template only to tag POS, so every span’s "labels"
  entry contains exactly one POS tag (e.g., "NOUN", "VERB").
* Each CSV row is treated as one sentence.  If your texts contain
  explicit sentence breaks, split them beforehand or add a splitter.

Output
------
# sent_id = <row number>
# text     = <original sentence>
1  FORM  form  UPOS  _  _  0  dep  _  _
2  ...
"""

import csv, json, re
from pathlib import Path

# Adjust these paths as needed. The input CSV should have columns "text" and "label".
INPUT  = Path("someFile.csv")
OUTPUT = Path("somePOSDataset.conllu")

TOKEN_REGEX = re.compile(r"\S+")


def load_labels(label_json: str):
    """Return dict { (start, end) : POS } for the row."""
    if not isinstance(label_json, str) or not label_json.strip():
        return {}
    spans = json.loads(label_json)
    return { (span["start"], span["end"]) : span["labels"][0]
             for span in spans if span.get("labels") }


with INPUT.open(newline="", encoding="utf-8") as f_in, OUTPUT.open("w", encoding="utf-8") as f_out:
    reader  = csv.DictReader(f_in)
    sent_id = 1

    for row in reader:
        text   = row["text"]
        labels = load_labels(row["label"])

        # If the label column is empty, warn and skip
        if not labels and any(row.values()):
            print(f"⚠️  Warning: no labels found for row {sent_id}: {text[:30]}...")
            continue
        if not text.strip():
            print(f"⚠️  Warning: empty text for row {sent_id}, skipping.")
            continue

        # write sentence-level comments
        f_out.write(f"# sent_id = {sent_id}\n")
        f_out.write(f"# text = {text}\n")

        # iterate tokens with character offsets
        for idx, match in enumerate(TOKEN_REGEX.finditer(text), start=1):
            start, end = match.span()
            form       = match.group()
            lemma      = form.lower()          # placeholder lemma
            upos       = labels.get((start, end), "_")

            conllu_cols = [
                str(idx),               # ID
                form,                   # FORM
                lemma,                  # LEMMA
                upos,                   # UPOS
                "_",                    # XPOS
                "_",                    # FEATS
                "0",                    # HEAD
                "dep",                  # DEPREL
                "_",                    # DEPS
                "_",                    # MISC
            ]
            f_out.write("\t".join(conllu_cols) + "\n")

        f_out.write("\n")               # blank line = sentence break
        sent_id += 1

print(f"Wrote {OUTPUT.resolve()}")
