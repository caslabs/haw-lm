"""Interactive command-line POS tagger powered by a BERT model fine-tuned on
the Universal Dependencies English Web Treebank (UD-EWT).

This script loads the tokenizer, model weights, and `label_map.json` found in
`MODEL_DIR`, then starts a REPL that prints the predicted part-of-speech (UD-17
tagset) and its soft-max confidence for each word you enter.

Requirements
------------
* Python 3.9+
* `torch >= 2.2`, `transformers >= 4.40`

A valid `MODEL_DIR` must contain:
    ├─ config.json
    ├─ pytorch_model.bin or model.safetensors
    ├─ tokenizer.json (+ vocab files)
    └─ label_map.json
"""

import json, os, torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch.nn.functional as F

# Refer back to `bert_pos_tagger_edewt_finetune.py` for model exportation procedure.
MODEL_DIR   = "bert-pos-ewt-finetuned"
MAX_LEN     = 128
IGNORE_LABEL = -100


# We load the model, tokenizer and label map.
print("Loading model from", MODEL_DIR, "...")
tok   = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()

# label map is already saved in `label_map.json` from the model exportation procedure
label_map_path = os.path.join(MODEL_DIR, "label_map.json")
id2label = (
    json.load(open(label_map_path))
    if os.path.exists(label_map_path)
    else model.config.id2label
)
display_mapping = {int(k): v for k, v in id2label.items()}

if all(isinstance(v, int) for v in display_mapping.values()):
    ud_tags = ["ADJ","ADP","ADV","AUX","CCONJ","DET","INTJ",
               "NOUN","NUM","PART","PRON","PROPN","PUNCT",
               "SCONJ","SYM","VERB","X"]         # 17 tags
    display_mapping = {i: tag for i, tag in enumerate(ud_tags)}


def predict_pos(text: str):
    """Return words, tags and confidence scores for one sentence."""
    words = text.strip().split()
    if not words:
        return [], [], []

    enc = tok(
        words,
        is_split_into_words=True,
        truncation=True,
        padding=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        logits = model(**enc).logits          # [1, seq_len, num_labels]
        probs  = F.softmax(logits, dim=-1)
        pred_ids = probs.argmax(-1)[0]        # first (and only) batch element

    word_ids = enc.word_ids()
    preds, confs = [], []
    seen_word = None
    for i, wid in enumerate(word_ids):
        if wid is None or wid == seen_word:
            continue                          # skip special tokens / sub-word pieces
        seen_word = wid
        pid = pred_ids[i].item()
        preds.append(display_mapping.get(pid, f"TAG_{pid}"))
        confs.append(probs[0, i, pid].item())

    return words, preds, confs

def pretty_print(words, tags, confs):
    print("\n" + "="*60)
    for ix, (w, t, c) in enumerate(zip(words, tags, confs), 1):
        print(f"{ix:2d}. {w:<15} → {t:<7} (p={c:.3f})")
    sent = " ".join(f"{w}/{t}" for w, t in zip(words, tags))
    print("-"*60)
    print("Tagged sentence:", sent)
    print("="*60 + "\n")

def main():
    print("="*72)
    print("INTERACTIVE POS TAGGER (fine-tuned bert-base-cased)")
    print("Tags:", ", ".join(sorted(set(display_mapping.values()))))
    print("Type empty line or Ctrl-C to quit.")
    print("="*72)
    while True:
        try:
            txt = input("\nEnter text: ").strip()
            if not txt:
                break
            words, tags, confs = predict_pos(txt)
            pretty_print(words, tags, confs)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print("Error:", e)

if __name__ == "__main__":
    main()
