from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from datasets import load_dataset

POS_DIR   = Path("../model_out/pos_distil")
CONLLU  = Path("../data/pos-makana/haw_pos_corpus.conllu")


# Load the POS tagging model and tokenizer
tok   = AutoTokenizer.from_pretrained(POS_DIR)
model = AutoModelForTokenClassification.from_pretrained(POS_DIR)
id2lab = model.config.id2label
print("id2label mapping:", id2lab, "\n")


# Load Samople Sentence from CONLLU
words, gold = [], []          # gold tags optional; useful for eyeballing
with open(CONLLU, encoding="utf-8") as f:
    for line in f:
        line = line.rstrip()
        if not line or line.startswith("#"):
            if words: break                  # end-of-sentence reached
            continue
        cols = line.split("\t")
        words.append(cols[1])                # FORM column
        gold.append(cols[3])                 # UPOS column

print("sample sentence :", " ".join(words))
print("gold UPOS (for reference):", gold)


# Tokenize the sample sentence and run through the model
# Note: we use `is_split_into_words=True` to ensure the model sees the
#       original words (not sub-tokens) for prediction.
#       This is important for POS tagging, as we want to predict tags for
#       whole words, not for sub-word pieces.
enc = tok(words, is_split_into_words=True, return_tensors="pt")
with torch.no_grad():
    logits = model(**enc).logits[0]
pred_ids = logits.argmax(dim=-1).tolist()

# Show the results
print("\nword           predicted")
print("-"*28)
seen = set()
for word_id, pred in zip(enc.word_ids(), pred_ids):
    if word_id is None or word_id in seen:      # skip [CLS]/[SEP] & sub-tokens
        continue
    print(f"{words[word_id]:<14} {id2lab[pred]}")
    seen.add(word_id)
