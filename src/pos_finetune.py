import json, random, yaml, numpy as np
from math import floor
from pathlib import Path
from collections import Counter
from conllu import parse
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    BertConfig, BertForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments, Trainer, TrainerCallback
)
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import accuracy_score, f1_score
from gensim.models import Word2Vec
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from gensim.models import Word2Vec
import mlflow
import os
import optuna
import datasets
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score
import json


GLOBAL_LABELS = [
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM",
    "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "VERB", "X", "dep"
]
GLOBAL_LABEL2ID = {label: i for i, label in enumerate(GLOBAL_LABELS)}
GLOBAL_ID2LABEL = {i: label for label, i in GLOBAL_LABEL2ID.items()}


class POSDataset(torch.utils.data.Dataset):
    """
    Custom dataset for loading POS-tagged data from JSON files.
    Each example in the dataset is expected to have a "tokens" list and a "tags" list.
    Parameters:
    - file: Path to the JSON file containing POS-tagged data.
    - vocab: Dictionary mapping tokens to their Word2Vec indices.
    - tag2idx: Dictionary mapping POS tags to their integer indices.
    """
    def __init__(self, file, vocab, tag2idx):
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.x = [[vocab.get(tok, vocab.get("<UNK>")) for tok in ex["tokens"]] for ex in data]
        self.y = [[tag2idx[tag] for tag in ex["tags"]] for ex in data]

    def __len__(self): return len(self.x)
    def __getitem__(self, i): return torch.tensor(self.x[i]), torch.tensor(self.y[i])

class Word2VecLSTMTagger(nn.Module):
    """
    A simple LSTM-based POS tagger using pre-trained Word2Vec embeddings.
    This model takes in token indices, embeds them using Word2Vec weights,
    and passes them through an LSTM layer followed by a linear layer for classification.
    Parameters:
    - embedding_weights: Pre-trained Word2Vec weights (numpy array).
    - hidden_dim: Number of hidden units in the LSTM layer.
    - output_dim: Number of POS tags (classes).
    - bidirectional: Whether to use a bidirectional LSTM.
    - dropout: Dropout rate for regularization.
    """
    def __init__(self, embedding_weights, hidden_dim, output_dim, bidirectional=True, dropout=0.0):
        super().__init__()
        vocab_size, embed_dim = embedding_weights.shape
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_weights), freeze=False)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), output_dim)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        """
        Forward pass through the model.
        Parameters:
        - x: Input tensor of shape (batch_size, seq_length) containing token indices.
        Returns:
        - out: Output tensor of shape (batch_size, seq_length, output_dim) containing log probabilities for each POS tag.
        """
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = self.dropout(out)
        return self.softmax(self.fc(out))

def collate_fn(batch):
    """
    Custom collate function to pad sequences in a batch.
    This function takes a batch of (input, target) pairs and pads them to the maximum sequence length in the batch.
    Parameters:
    - batch: List of tuples where each tuple contains (input_tensor, target_tensor).
    Returns:
    - Padded input tensor of shape (batch_size, max_seq_length).
    - Padded target tensor of shape (batch_size, max_seq_length) with -100
    """
    x, y = zip(*batch)
    return pad_sequence(x, batch_first=True), pad_sequence(y, batch_first=True, padding_value=-100)

def objective(trial, base_cfg):
    """
    Objective function for hyperparameter optimization of Word2Vec + LSTM POS tagging model.
    This function is called by Optuna to evaluate different hyperparameter combinations.
    """
    # Load vocab + embeddings
    w2v = Word2Vec.load(os.path.join(base_cfg["checkpoint_path"], "word2vec.model"))
    embedding_weights = w2v.wv.vectors
    vocab = w2v.wv.key_to_index
    if "<UNK>" not in vocab:
        vocab["<UNK>"] = len(vocab)
        unk_vec = torch.mean(torch.tensor(embedding_weights), dim=0).numpy()
        embedding_weights = np.vstack([embedding_weights, unk_vec])

    # Load tags
    with open("data/pos_train.json") as f:
        tag_set = sorted({tag for ex in json.load(f) for tag in ex["tags"]})
    tag2idx = {tag: i for i, tag in enumerate(tag_set)}

    # Read JSON data for training and validation
    train_ds = POSDataset("data/pos_train.json", vocab, tag2idx)
    val_ds = POSDataset("data/pos_validation.json", vocab, tag2idx)

    # Hyperparameters from Optuna trial
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256, 384, 512])
    lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
    bidirectional = trial.suggest_categorical("bidirectional", [True, False])
    epochs = trial.suggest_int("epochs", 5, 30)
    dropout = trial.suggest_float("dropout", 0.1, 0.4)
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.3)
    optimizer_name = trial.suggest_categorical("optimizer", ["adam", "adamw", "sgd"])

    # Data loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Initialize model
    model = Word2VecLSTMTagger(
        embedding_weights,
        hidden_dim,
        output_dim=len(tag2idx),
        bidirectional=bidirectional,
    )

    # Optimizer selection for hyperparameter tuning
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Loss function
    # Using NLLLoss with ignore_index=-100 for padding in the target sequences
    loss_fn = nn.NLLLoss(ignore_index=-100)
    model.train()
    for _ in range(epochs):
        for xb, yb in train_loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = loss_fn(out.view(-1, out.shape[-1]), yb.view(-1))
            loss.backward()
            optimizer.step()

    # Evaluate the model on the validation set
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            out = model(xb).argmax(-1)
            mask = yb != -100
            all_preds.extend(out[mask].tolist())
            all_labels.extend(yb[mask].tolist())


    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return 1 - f1  # Minimize 1 - F1 (maximize F1)


def word2vec_lstm_pos_classifier(cfg):
    """
    Train a Word2Vec + LSTM model on POS-tagged data
    """

    # Load Word2Vec
    model_path = os.path.join(cfg["checkpoint_path"], "word2vec.model")
    w2v = Word2Vec.load(model_path)
    embedding_weights = w2v.wv.vectors
    vocab = w2v.wv.key_to_index
    if "<UNK>" not in vocab:
        vocab["<UNK>"] = len(vocab)
        unk_vector = torch.mean(torch.tensor(embedding_weights), dim=0).numpy()
        embedding_weights = np.vstack([embedding_weights, unk_vector])

    # Read JSON data
    train_file = "data/pos_train.json"
    valid_file = "data/pos_validation.json"

    with open(train_file, "r") as f:
        tag_set = sorted({tag for ex in json.load(f) for tag in ex["tags"]})
    tag2idx = {tag: i for i, tag in enumerate(tag_set)}

    train_dataset = POSDataset(train_file, vocab, tag2idx)
    valid_dataset = POSDataset(valid_file, vocab, tag2idx)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # Initialize model
    model = Word2VecLSTMTagger(embedding_weights, hidden_dim=128, output_dim=len(tag2idx))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.NLLLoss(ignore_index=-100)

    # Train
    model.train()
    for epoch in range(5):
        total_loss = 0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = loss_fn(logits.view(-1, logits.shape[-1]), y_batch.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    # Evaluate
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x_batch, y_batch in valid_loader:
            logits = model(x_batch)
            preds = logits.argmax(-1)
            mask = y_batch != -100
            all_preds.extend(preds[mask].tolist())
            all_labels.extend(y_batch[mask].tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    print(f"Word2Vec+LSTM POS Accuracy: {acc:.4f}")
    print(f"Word2Vec+LSTM POS F1 Score: {f1:.4f}")
    return {"eval_accuracy": acc,
            "eval_f1": f1}


def run_hpo_for_word2vec(cfg):
    """
    Run hyperparameter optimization for Word2Vec + LSTM POS tagging model using Optuna.
    This function will create a study, optimize the objective function,
    and return the best trial metrics.
    """
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, cfg), n_trials=10)

    print("Best Word2Vec trial:")
    print(study.best_trial)

    # Capture the best parameters
    best_params = study.best_trial.params

    # Load vocab + embeddings
    w2v = Word2Vec.load(os.path.join(cfg["checkpoint_path"], "word2vec.model"))
    embedding_weights = w2v.wv.vectors
    vocab = w2v.wv.key_to_index
    if "<UNK>" not in vocab:
        vocab["<UNK>"] = len(vocab)
        unk_vec = torch.mean(torch.tensor(embedding_weights), dim=0).numpy()
        embedding_weights = np.vstack([embedding_weights, unk_vec])

    # Load tags
    with open("data/pos_train.json") as f:
        tag_set = sorted({tag for ex in json.load(f) for tag in ex["tags"]})
    tag2idx = {tag: i for i, tag in enumerate(tag_set)}

    # Read JSON data for training and validation
    train_ds = POSDataset("data/pos_train.json", vocab, tag2idx)
    val_ds = POSDataset("data/pos_validation.json", vocab, tag2idx)

    # Hyperparameters from Optuna trial
    batch_size = best_params.get("batch_size", 128)
    hidden_dim = best_params.get("hidden_dim", 256)
    lr = best_params.get("lr", 0.001)
    bidirectional = best_params.get("bidirectional", True)
    epochs = best_params.get("epochs", 10)
    dropout =  best_params.get("dropout", 0.1)
    weight_decay = best_params.get("weight_decay", 0.0)
    optimizer_name = best_params.get("optimizer", "adam")

    # Data loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Initialize model
    model = Word2VecLSTMTagger(
        embedding_weights,
        hidden_dim,
        output_dim=len(tag2idx),
        bidirectional=bidirectional,
    )

    # Optimizer selection for hyperparameter tuning
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Loss function
    loss_fn = nn.NLLLoss(ignore_index=-100)
    model.train()
    for _ in range(epochs):
        for xb, yb in train_loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = loss_fn(out.view(-1, out.shape[-1]), yb.view(-1))
            loss.backward()
            optimizer.step()

    # Evaluate the model on the validation set
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            out = model(xb).argmax(-1)
            mask = yb != -100
            all_preds.extend(out[mask].tolist())
            all_labels.extend(yb[mask].tolist())


    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")

    # Log metrics to MLflow
    mlflow.log_metric("eval_accuracy", acc)
    mlflow.log_metric("eval_f1", f1)

    return {
        "eval_accuracy": acc,
        "eval_f1": f1
    }


class MLflowCallback(TrainerCallback):
    """
    A simple callback to log metrics to MLflow during training.
    """
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(k, float(v), step=state.global_step)


def read_conllu(path):
    """
    Read a CoNLL-U file and return a list of sentences.
    """
    with open(path, "r", encoding="utf-8") as f:
        return parse(f.read())


def split_conllu(path, split=(0.8, 0.1, 0.1), seed=42):
    """
    Split a CoNLL-U file into train, validation, and test sets.
    """
    random.seed(seed)
    sents = read_conllu(path)
    total  = len(sents)
    if total < 3:
        raise ValueError("Need ≥3 sentences for train/val/test")

    random.shuffle(sents)
    n_train = max(1, floor(split[0] * total))
    n_val   = max(1, floor(split[1] * total))
    n_test  = max(1, total - n_train - n_val)

    def to_dicts(sentences):
        """
        Convert a list of CoNLL-U sentences to a list of dictionaries
        """
        return [
            {"tokens": [t["form"] for t in s],
             "upos":   [t["upos"]  for t in s]}
            for s in sentences
        ]

    return datasets.DatasetDict({
        "train":       datasets.Dataset.from_list(to_dicts(sents[:n_train])),
        "validation":  datasets.Dataset.from_list(to_dicts(sents[n_train:n_train+n_val])),
        "test":        datasets.Dataset.from_list(to_dicts(sents[n_train+n_val:]))
    })


def build_tokenizer(cfg):
    """
    Build the tokenizer based on the configuration.
    If `tokenizer_path` is specified, use that. Otherwise, use the checkpoint path.
    """
    if cfg.get("tokenizer_path"):
        return AutoTokenizer.from_pretrained(cfg["tokenizer_path"])
    return AutoTokenizer.from_pretrained(cfg["checkpoint_path"])


def build_pos_model(cfg, num_labels, label2id, id2label):
    """
    Build the POS tagging model based on the configuration.
    If `train_from_scratch` is True, create a new model based on the config.
    Otherwise, load a pre-trained model from HuggingFace.
    If `num_labels`, `id2label`, and `label2id` are provided
    """

    # Check label mapping consistency
    assert num_labels == len(label2id) == len(id2label), "Label mapping inconsistency"

    # Check if we need to train from scratch
    if cfg.get("train_from_scratch", False):
        # From-scratch BERT (same dims as pre-train cfg)
        return AutoModelForTokenClassification.from_pretrained(
            cfg["checkpoint_path"],
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            local_files_only=True
        )

    # Load pre-trained model with specified labels
    return AutoModelForTokenClassification.from_pretrained(
        cfg["checkpoint_path"],
        num_labels=num_labels,                # ← always 17
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True         # ← now no mismatch
    )

def export_pos_json(dataset: datasets.DatasetDict, out_dir="data"):
    """
    Export 'train' and 'validation' splits of a POS-tagged DatasetDict
    to JSON files expected by Word2Vec+LSTM POS classifier.
    Each file will contain a list of dicts: {"tokens": [...], "tags": [...]}
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def to_dicts(split):
        return [{"tokens": row["tokens"], "tags": row["upos"]} for row in dataset[split]]

    for split in ("train", "validation"):
        with open(out_dir / f"pos_{split}.json", "w", encoding="utf-8") as f:
            json.dump(to_dicts(split), f, ensure_ascii=False, indent=2)

    print("Exported POS-tagged JSON files for Word2Vec-LSTM.")


def load_tfidf_data(train_file="data/pos_train.json", val_file="data/pos_validation.json"):
    """
    Load TF-IDF data from JSON files for training and validation.
    Each file should contain a list of dictionaries with "tokens" and "tags" keys.
    """
    def load(file):
        with open(file, encoding="utf-8") as f:
            data = json.load(f)
        return [x["tokens"] for x in data], [x["tags"] for x in data]

    return *load(train_file), *load(val_file)


def objective_tfidf(trial, cfg):
    """
    Objective function for hyperparameter optimization of TF-IDF + Logistic Regression model.
    This function is called by Optuna to evaluate different hyperparameter combinations.
    """
    X_train_raw, y_train_raw, X_val_raw, y_val_raw = load_tfidf_data()

    # Hyperparameters
    max_features = trial.suggest_int("max_features", 1000, 5000)
    ngram_range = trial.suggest_categorical("ngram_range", [(1, 1), (1, 2)])
    C = trial.suggest_float("C", 0.01, 10.0, log=True)

    # Flatten and align
    X_train_tokens, y_train_tokens = [], []
    for tokens, tags in zip(X_train_raw, y_train_raw):
        if len(tokens) != len(tags):
            continue
        for t, tag in zip(tokens, tags):
            if t.strip():  # skip empty tokens
                X_train_tokens.append(t)
                y_train_tokens.append(tag)

    X_val_tokens, y_val_tokens = [], []
    for tokens, tags in zip(X_val_raw, y_val_raw):
        if len(tokens) != len(tags):
            continue
        for t, tag in zip(tokens, tags):
            if t.strip():
                X_val_tokens.append(t)
                y_val_tokens.append(tag)

    # Encode tags
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train_tokens)
    y_val_encoded = le.transform(y_val_tokens)

    # Build pipeline
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words=None,  # very important: Hawaiian data
    )
    clf = LogisticRegression(C=C, max_iter=200, solver="liblinear")
    pipe = Pipeline([("tfidf", vectorizer), ("clf", clf)])

    # Fit
    pipe.fit(X_train_tokens, y_train_encoded)
    preds = pipe.predict(X_val_tokens)

    acc = accuracy_score(y_val_encoded, preds)
    f1 = f1_score(y_val_encoded, preds, average="macro")
    return 1 - f1


def run_hpo_for_tfidf(cfg):
    """
    Run hyperparameter optimization for TF-IDF + Logistic Regression model using Optuna.
    This function will create a study, optimize the objective function,
    and return the best trial metrics.
    """
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective_tfidf(trial, cfg), n_trials=20)

    print("Best TF-IDF Trial:")
    print(study.best_trial)

    # Capture the best parameters
    best_params = study.best_trial.params

    # Redo training with best parameters
    X_train_raw, y_train_raw, X_val_raw, y_val_raw = load_tfidf_data()

    # Hyperparameters
    max_features = best_params["max_features"]
    ngram_range = best_params["ngram_range"]
    C = best_params["C"]

    # Flatten and align
    X_train_tokens, y_train_tokens = [], []
    for tokens, tags in zip(X_train_raw, y_train_raw):
        if len(tokens) != len(tags):
            continue
        for t, tag in zip(tokens, tags):
            if t.strip():  # skip empty tokens
                X_train_tokens.append(t)
                y_train_tokens.append(tag)

    X_val_tokens, y_val_tokens = [], []
    for tokens, tags in zip(X_val_raw, y_val_raw):
        if len(tokens) != len(tags):
            continue
        for t, tag in zip(tokens, tags):
            if t.strip():
                X_val_tokens.append(t)
                y_val_tokens.append(tag)

    # Encode tags
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train_tokens)
    y_val_encoded = le.transform(y_val_tokens)

    # Build pipeline
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words=None,
    )
    clf = LogisticRegression(C=C, max_iter=200, solver="liblinear")
    pipe = Pipeline([("tfidf", vectorizer), ("clf", clf)])

    # Fit
    pipe.fit(X_train_tokens, y_train_encoded)
    preds = pipe.predict(X_val_tokens)

    acc = accuracy_score(y_val_encoded, preds)
    f1 = f1_score(y_val_encoded, preds, average="macro")


    # Log metrics to MLflow
    mlflow.log_metric("eval_accuracy", acc)
    mlflow.log_metric("eval_f1", f1)

    return {
        "eval_accuracy": acc,
        "eval_f1": f1
    }


def tfidf_linear_pos_classifier(cfg):
    """
    TODO: Train a TF-IDF + Logistic Regression model on POS-tagged data.
    """
    acc = 1
    f1 = 1
    return {"eval_accuracy": acc, "eval_f1": f1}
def objective_transformer(trial, cfg):
    # Suggest hyperparams
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
    batch_size = trial.suggest_categorical("batch_size", [4, 8, 16])
    num_epochs = trial.suggest_int("epochs", 3, 10)
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.3)

    # Override config for trial
    cfg = cfg.copy()
    cfg["learning_rate"] = learning_rate
    cfg["batch_size"] = batch_size
    cfg["epochs"] = num_epochs
    cfg["weight_decay"] = weight_decay

    # Re-run training with current trial config
    metrics = run_transformer_pos(cfg)
    return 1 - metrics["eval_f1"]

def run_transformer_pos(cfg):
    conllu_file = cfg.get("conllu_path", "data/pos-makana/v1/makana_v1.conllu")

    dataset = split_conllu(
        conllu_file,
        split=(0.8, 0.1, 0.1),
        seed=int(cfg["seed"])
    )

    flat_tags = [t for row in dataset["train"] for t in row["upos"]]
    print("Train tag distribution:", Counter(flat_tags))

    tag_set = GLOBAL_LABELS
    label2id = GLOBAL_LABEL2ID
    id2label = GLOBAL_ID2LABEL


    def add_labels(ex):
        # If there's any '_' in the labels, skip the example
        if "_" in ex["upos"]:
            raise ValueError(f"Encountered '_' in labels for example: {ex}")

        ex["labels"] = [label2id[t] for t in ex["upos"]]
        return ex


    dataset = dataset.map(add_labels)

    # Tokenizer & alignment
    tok = build_tokenizer(cfg)

    def tok_align(ex):
        enc = tok(ex["tokens"],
                  is_split_into_words=True,
                  truncation=True,
                  padding="max_length",
                  max_length=int(cfg["max_length"]))
        word_ids = enc.word_ids()
        aligned = []
        prev = None
        for wid in word_ids:
            if wid is None: aligned.append(-100)
            elif wid != prev:
                aligned.append(ex["labels"][wid]); prev = wid
            else:
                aligned.append(ex["labels"][wid])
        enc["labels"] = aligned
        return enc

    if "train" not in dataset or len(dataset["train"]) == 0 or "tokens" not in dataset["train"].column_names:
        raise ValueError(f"Expected 'train' split with 'tokens' column. Got splits: {dataset.keys()} and columns: {dataset['train'].column_names if 'train' in dataset else 'N/A'}")

    ds_tok = dataset.map(tok_align, batched=False)
    required_cols = {"tokens", "upos"}
    for split in ["train", "validation"]:
        if not required_cols.issubset(set(ds_tok[split].column_names)):
            raise ValueError(
                f"{split} split is missing required columns. "
                f"Required: {required_cols}, Got: {ds_tok[split].column_names}"
            )

    # Optionally check test set too, but skip if it's empty
    if ds_tok.get("test") and len(ds_tok["test"]) > 0:
        if not required_cols.issubset(set(ds_tok["test"].column_names)):
            raise ValueError(
                f"test split is missing required columns. "
                f"Required: {required_cols}, Got: {ds_tok['test'].column_names}"
            )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        valid_preds, valid_labels = [], []
        for pred_seq, label_seq in zip(preds, labels):
            for p, l in zip(pred_seq, label_seq):
                if l != -100:
                    valid_preds.append(p)
                    valid_labels.append(l)
        acc = accuracy_score(valid_labels, valid_preds)
        f1 = f1_score(valid_labels, valid_preds, average="macro")
        mlflow.log_metric("eval_accuracy", acc)
        mlflow.log_metric("eval_f1", f1)
        return {"eval_accuracy": acc, "eval_f1": f1}

    # ─── POS-specific hyper-params pulled from cfg ─────────────
    IGNORE            = int(cfg.get("ignore_label", -100))
    POS_BS            = int(cfg.get("pos_batch_size", 32))
    POS_LR            = float(cfg.get("pos_learning_rate", 3e-5))
    POS_EPH           = int(cfg.get("pos_num_epochs", 5))
    POS_EVAL_STRAT    = cfg.get("pos_eval_strategy", "epoch")   # "epoch" | "steps"

    # ─── HF TrainingArguments (everything in one block) ────────
    args = TrainingArguments(
        output_dir                   = cfg["pos_output_dir"],
        per_device_train_batch_size  = POS_BS,
        per_device_eval_batch_size   = POS_BS,
        num_train_epochs             = POS_EPH,
        learning_rate                = POS_LR,
        weight_decay                 = cfg.get("weight_decay", 0.01),

        eval_strategy                = POS_EVAL_STRAT,   # ← renamed to official arg
        save_strategy                = "no",             # one final save later
        logging_strategy             = "steps",
        logging_steps                = 1,

        report_to                    = "none",
        seed                         = int(cfg["seed"]),
    )


    trainer = Trainer(
            model_init      = lambda: build_pos_model(cfg, len(tag_set),
                                                    label2id, id2label),
            args            = args,
            train_dataset   = ds_tok["train"],
            eval_dataset    = ds_tok["validation"],
            tokenizer       = tok,
            data_collator   = DataCollatorForTokenClassification(tokenizer=tok),
            compute_metrics = compute_metrics,
            callbacks       = [MLflowCallback()],
        )

    if cfg.get("search", False):
        def hp_space(trial):
            return {
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True),
                "weight_decay":  trial.suggest_float("weight_decay", 0.0, 0.3),
                "num_train_epochs": trial.suggest_int("num_train_epochs", 3, 10),
                "per_device_train_batch_size": trial.suggest_categorical(
                    "per_device_train_batch_size", [4, 8, 16]),
            }

        best_run = trainer.hyperparameter_search(
            direction        = "maximize",   # we want highest F1
            backend          = "optuna",
            hp_space         = hp_space,
            n_trials         = 10,
            compute_objective= lambda m: m["eval_f1"],
        )
        for k, v in best_run.hyperparameters.items():
            setattr(args, k, v)

        # Train with best hyperparameters
        trainer = Trainer(
            model_init      = lambda: build_pos_model(cfg, len(tag_set),
                                                    label2id, id2label),
            args            = args,
            train_dataset   = ds_tok["train"],
            eval_dataset    = ds_tok["validation"],
            tokenizer       = tok,
            data_collator   = DataCollatorForTokenClassification(tokenizer=tok),
            compute_metrics = compute_metrics,
            callbacks       = [MLflowCallback()],
        )
        trainer.train()
        metrics = trainer.evaluate()
        trainer.save_model(cfg["pos_output_dir"])
        return metrics

    # If no HPO, just train once as before
    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(cfg["pos_output_dir"])
    return metrics


def run(cfg: dict) -> dict:
    """
    cfg comes straight from params.yaml (one experiment entry)
    Returns evaluation metrics (accuracy) for MLflow logging.
    """

    # TRADITIONAL MODELS
    if cfg.get("model_type") == "word2vec":
        dataset = split_conllu("data/pos-makana/v1/makana_v1.conllu",
                               split=(0.8, 0.1, 0.1),
                               seed=int(cfg["seed"]))
        export_pos_json(dataset)
        if cfg.get("search", False):
            return run_hpo_for_word2vec(cfg)
        else:
            return word2vec_lstm_pos_classifier(cfg)

    elif cfg.get("model_type") == "tfidf":
        dataset = split_conllu("data/pos-makana/v1/makana_v1.conllu",
                                split=(0.8, 0.1, 0.1),
                                seed=int(cfg["seed"]))
        export_pos_json(dataset)
        if cfg.get("search", False):
            return run_hpo_for_tfidf(cfg)
        else:
            return tfidf_linear_pos_classifier(cfg)
    else:
        return run_transformer_pos(cfg)
