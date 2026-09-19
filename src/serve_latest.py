from pathlib import Path
import gradio as gr
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import re

# Specify the base directory and model directories
# TODO: consider using environment variables or command-line arguments for flexibility
BASE_DIR = Path(__file__).resolve().parent
MLM_DIR  = (BASE_DIR / "../quick-haw-mlm/checkpoint-300").resolve()
POS_DIR  = (BASE_DIR / "../model_out/pos_distil").resolve()

# Load the Masked Language Model (MLM) pipeline
fill_mask = pipeline(
    "fill-mask",
    model=str(MLM_DIR),
    tokenizer=str(MLM_DIR),
    top_k=5,
    device_map="auto",
)

# Retrieve the latest checkpoint for the MLM
try:
    tok_pos  = AutoTokenizer.from_pretrained(POS_DIR)
    mod_pos  = AutoModelForTokenClassification.from_pretrained(POS_DIR)
    id2label = mod_pos.config.id2label
    mod_pos.eval()
    has_pos  = True
except Exception as e:
    print(f"⚠️  POS model not available ({e}); POS tab will be hidden.")
    tok_pos = mod_pos = id2label = None
    has_pos = False

def do_mask(sentence: str):
    """
    Fill the [MASK] token in the given sentence using the MLM pipeline.
    """
    if "[MASK]" not in sentence:
        return [["(include a [MASK] token)", ""]]
    outs = fill_mask(sentence)
    return [[o["token_str"], f"{o['score']:.3f}"] for o in outs]

def do_pos(sentence: str):
    """
    Tag the parts of speech (POS) for each token in the given sentence using
    the POS model.
    1) Tokenizes the sentence using a regex that keeps punctuation separate.
    2) Encodes the tokens using the POS tokenizer.
    3) Runs the POS model to get predictions.
    4) Returns a list of [token, POS tag] pairs.
    5) Skips special tokens and sub-token continuations.
    6) Returns a list of [token, POS tag] pairs for the original words
    """
    if mod_pos is None:
        return "❌ POS tagger not available."

    # simple regex tokenizer → keeps punctuation separate
    tokens = re.findall(r"\w+ʻ?\w+|['ʻ\w]+|[^\w\s]", sentence, flags=re.UNICODE)

    # encode pre-split tokens
    enc = tok_pos(tokens, is_split_into_words=True, return_tensors="pt")

    # run the POS model
    with torch.no_grad():
        preds = mod_pos(**enc).logits[0].argmax(dim=-1).tolist()

    # collect results, skipping special tokens and sub-token continuations
    rows, seen_word = [], set()
    for word_id, pred in zip(enc.word_ids(), preds):
        if word_id is None or word_id in seen_word:
            continue
        rows.append([tokens[word_id], id2label[pred]])
        seen_word.add(word_id)
    return rows


# Gradio UI
with gr.Blocks(title="Towards Olelo Hawaii Language Model") as demo:
    gr.Markdown("## Towards Olelo Hawaii Language Model")

    # Masked Language Model Tab
    with gr.Tab("Masked Language Model"):
        txt_mlm = gr.Textbox(
            label="Sentence (include [MASK])",
            placeholder="ʻO ka ʻāina [MASK] ke kanaka."
        )
        out_mlm = gr.Dataframe(headers=["Prediction", "p"], interactive=False)
        gr.Button("Predict").click(do_mask, txt_mlm, out_mlm)

    # POS Tagger Tab
    if has_pos:
        with gr.Tab("POS Tagger"):
            txt_pos = gr.Textbox(label="Sentence",
                                 placeholder="ʻO ke kumu nui kēia.")
            out_pos = gr.Dataframe(headers=["Token", "POS"], interactive=False)
            gr.Button("Tag").click(do_pos, txt_pos, out_pos)
    else:
        gr.Markdown("> POS tagger not available for these checkpoints.")


# Launch the Gradio app
if __name__ == "__main__":
    demo.launch(server_port=7860, show_api=False, share=True)
