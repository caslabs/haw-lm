import json
import os
import re
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForTokenClassification


def load_model(model_dir, max_len=128):
    """
    Load model, tokenizer, and label map from given directory.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    label_map_path = os.path.join(model_dir, "label_map.json")
    id2label = (
        json.load(open(label_map_path))
        if os.path.exists(label_map_path)
        else model.config.id2label
    )


    # Convert string keys to ints if needed
    if all(k.isdigit() for k in id2label.keys()):
        display_mapping = {int(k): v for k, v in id2label.items()}
    else:
        display_mapping = {v: k for k, v in id2label.items()}

    # Fallback: use Universal POS tags if mapping is invalid
    if all(isinstance(v, int) for v in display_mapping.values()):
        ud_tags = [
            "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ",
            "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT",
            "SCONJ", "SYM", "VERB", "X"
        ]
        display_mapping = {i: tag for i, tag in enumerate(ud_tags)}

    return tokenizer, model, device, display_mapping, max_len


def predict_pos(text, tokenizer, model, device, display_mapping, max_len):
    """
    Predict POS tags for the input text using the provided tokenizer and model.
    This function tokenizes the input text, runs it through the model, and
    returns the predicted tags along with their confidence scores.
    """
    words = re.findall(r"\w+|\S", text.strip())
    if not words:
        return [], [], []

    enc = tokenizer(
        words,
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        logits = model(**enc).logits
        probs = F.softmax(logits, dim=-1)
        pred_ids = probs.argmax(-1)[0]

    word_ids = enc.word_ids()
    preds, confs = [], []
    seen_word = None
    for i, wid in enumerate(word_ids):
        if wid is None or wid == seen_word:
            continue
        seen_word = wid
        pid = pred_ids[i].item()
        preds.append(display_mapping.get(pid, f"TAG_{pid}"))
        confs.append(probs[0, i, pid].item())

    return words, preds, confs


def pretty_print(words, tags, confs):
    """
    Pretty-print the predicted POS tags with their confidence scores.
    """
    if not words:
        print("No input provided.")
        return

    print("\n" + "=" * 60)
    for ix, (w, t, c) in enumerate(zip(words, tags, confs), 1):
        print(f"{ix:2d}. {w:<15} → {t:<7} (p={c:.3f})")
    sent = " ".join(f"{w}/{t}" for w, t in zip(words, tags))
    print("-" * 60)
    print("Tagged sentence:", sent)
    print("=" * 60 + "\n")


def run(cfg):
    """
    Entry point for interactive POS tagger using config.
    """
    tokenizer, model, device, display_mapping, max_len = load_model(
        cfg["pos_model_dir"], max_len=cfg.get("max_length", 128)
    )

    print("=" * 72)
    print("INTERACTIVE POS TAGGER (Fine-tuned model)")
    print("Tags:", ", ".join(sorted(set(display_mapping.values()))))
    print("Type a sentence to tag, or /exit to quit.")
    print("=" * 72)

    while True:
        try:
            txt = input("\nEnter text: ").strip()
            if txt.lower() == "/exit":
                print("Exiting...")
                break
            words, tags, confs = predict_pos(txt, tokenizer, model, device, display_mapping, max_len)
            pretty_print(words, tags, confs)
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print("Error:", e)
