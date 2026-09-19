from typing import List, Dict, Optional, Any
import re
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
import sys, pathlib
import json
import string
import unicodedata
import shutil
import atexit
import glob

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

class HawaiianPOSModel(LabelStudioMLBase):
    """Hawaiian POS Tagger ML Backend model for Label Studio
    """
    # Constants for active learning retraining
    AL_MIN_BATCH = 25
    AL_MODEL_DIR = Path("/data2/jeraldy/haw-lm/distilbert_haw_pos_active_learning")
    AL_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Keep track of model versions (only keep 2)
    MAX_MODEL_VERSIONS = 2

    # Shared pool file that persists across instances
    AL_POOL_FILE = AL_MODEL_DIR / "current_pool.json"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Load existing pool from disk if it exists
        self._load_pool()

    def _load_pool(self):
        """Load the active learning pool from disk"""
        try:
            if self.AL_POOL_FILE.exists():
                with self.AL_POOL_FILE.open("r", encoding="utf-8") as f:
                    self._al_pool = json.load(f)
                print(f"[load_pool] Loaded {len(self._al_pool)} items from disk")
            else:
                self._al_pool = []
                print("[load_pool] Starting with empty pool")
        except Exception as e:
            print(f"[load_pool] Failed to load pool: {e}")
            self._al_pool = []

    def _save_pool(self):
        """Save the active learning pool to disk"""
        try:
            with self.AL_POOL_FILE.open("w", encoding="utf-8") as f:
                json.dump(self._al_pool, f, ensure_ascii=False, indent=2)
            print(f"[save_pool] Saved {len(self._al_pool)} items to disk")
        except Exception as e:
            print(f"[save_pool] Failed to save pool: {e}")

    def clear_pool(self):
        """Clear the active learning pool and remove from disk"""
        self._al_pool = []
        try:
            if self.AL_POOL_FILE.exists():
                self.AL_POOL_FILE.unlink()
                print("[clear_pool] Pool cleared from disk")
        except Exception as e:
            print(f"[clear_pool] Failed to clear pool file: {e}")

    def setup(self):
        """Configure any parameters of your model here
        """
        self.set("model_version", "hawaiian_pos_v1")

        # Configure paths and model loading
        BASE_DIR = Path(__file__).resolve().parent
        POS_DIR  = (BASE_DIR / "../model_out/pos_distil").resolve()

        # Note: Pool now persists across instances and loads from disk
        self._load_pool()
        print(f"[setup] Loaded AL pool with {len(self._al_pool)} items")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(POS_DIR)

            # Load full 17-tag label2id
            label_path = (POS_DIR / "label2id.json")
            print(f"[setup] Looking for label2id.json at: {label_path}")
            with open(label_path, "r", encoding="utf-8") as f:
                label2id = json.load(f)
            id2label = {v: k for k, v in label2id.items()}

            self.model = AutoModelForTokenClassification.from_pretrained(
                POS_DIR,
                num_labels=len(label2id),
                id2label=id2label,
                label2id=label2id,
                ignore_mismatched_sizes=True   # allows loading even if model only uses subset
            ).eval()
            self.id2label = id2label

            self.model_name_or_path = str(POS_DIR)
            print(f"Model labels: {self.id2label}")
        except Exception as e:
            raise RuntimeError(f"Failed to load POS model: {e}")

        # Tokenization regex for Hawaiian text - improved to handle punctuation properly
        self._tok_re = re.compile(r"\w+ʻ?\w*|['ʻ\w]+|[^\w\s]", re.UNICODE)

        # Label Studio configuration
        self.from_name = "label"  # must match your Label Studio config
        self.to_name = "text"     # must match your Label Studio config

    def is_punctuation(self, text: str) -> bool:
        """Check if text is punctuation using Unicode categories"""
        if not text:
            return False

        # Check each character in the text
        for char in text:
            category = unicodedata.category(char)
            # Unicode categories for punctuation: Pc, Pd, Pe, Pf, Pi, Po, Ps
            # Unicode categories for symbols: Sc, Sk, Sm, So
            if not (category.startswith('P') or category.startswith('S')):
                return False
        return True

    def get_spans(self, text: str) -> List[Dict[str, Any]]:
        """
        Get POS predictions and return in GLiNER-style format
        """
        tokens = self._tok_re.findall(text)

        # Debug: print tokens to see how they're being split
        print(f"[get_spans] Tokens: {tokens}")

        enc = self.tokenizer(tokens, is_split_into_words=True, return_tensors="pt")

        with torch.no_grad():
            logits = self.model(**enc).logits[0]
            preds = logits.argmax(-1).tolist()
            # Get confidence scores using softmax
            probs = torch.softmax(logits, dim=-1)
            confidences = probs.max(-1).values.tolist()

        spans, seen = [], set()
        char_offset = 0

        for wid, pred, conf in zip(enc.word_ids(), preds, confidences):
            if wid is None or wid in seen:
                continue

            word = tokens[wid]

            # Find the actual position in the text more accurately
            start = text.find(word, char_offset)
            if start == -1:  # fallback if not found
                start = char_offset
            end = start + len(word)
            char_offset = end

            # Map the label
            original_label = self.id2label[pred]

            # Special handling for punctuation using the helper function
            if self.is_punctuation(word):
                # If it's pure punctuation and not already labeled as PUNCT
                if original_label not in ['PUNCT', 'PERIOD', 'COMMA', 'QUESTION', 'EXCLAMATION']:
                    print(f"[get_spans] Punctuation '{word}' labeled as '{original_label}', should be PUNCT")
                    # You might want to override here if you have a PUNCT label
                    # original_label = 'PUNCT'  # uncomment if you want to force this

            spans.append({
                'start': start,
                'end': end,
                'text': word,
                'label': original_label,
                'score': conf
            })

            seen.add(wid)

        return spans

    def convert_to_ls_annotation(self, prediction: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert from POS model output format to Label Studio annotation format
        """
        results = []
        sent_preds = []

        for ent in prediction:
            label = [ent['label']]
            if label:
                score = ent['score']
                sent_preds.append({
                    'from_name': self.from_name,
                    'to_name': self.to_name,
                    'type': 'labels',
                    "value": {
                        "start": ent['start'],
                        "end": ent['end'],
                        "text": ent['text'],
                        "labels": label
                    },
                    "score": round(score, 4)
                })

        # Add minimum of certainty scores of entities in sentence for active learning use
        score = min([p['score'] for p in sent_preds]) if sent_preds else 0.0

        results.append({
            "result": sent_preds,
            "score": score,
            "model_version": self.get("model_version")
        })

        print(results)

        return results

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        """Run POS prediction on Hawaiian text
        :param tasks: Label Studio tasks in JSON format
        :param context: Label Studio context in JSON format
        :return model_response: ModelResponse with predictions
        """
        print(f"Using model: {self.model_name_or_path}")
        print(f'''\
        Run POS prediction on {len(tasks)} tasks
        Received context: {context}
        Project ID: {self.project_id}
        Label config: {self.label_config}
        Model version: {self.get("model_version")}''')

        predictions = []

        for task in tasks:
            # Extract text from task data
            text = task.get('data', {}).get('text', '')

            if not text:
                print(f"No text found in task: {task}")
                continue

            print(f"Processing text: {text[:50]}...")

            # Get POS predictions
            spans = self.get_spans(text)

            # Convert to Label Studio format
            ls_predictions = self.convert_to_ls_annotation(spans)

            # Add to predictions list
            predictions.extend(ls_predictions)

        print(f"Generated {len(predictions)} predictions")
        return ModelResponse(predictions=predictions)

    def fit(self, event, data, **kwargs):
        """
        Called by Label Studio after *every* annotation-related event.
        Collect rows → periodically retrain → hot-reload weights.
        """
        print(f"[fit] Event={event}  len(data)={len(data) if hasattr(data,'__len__') else '?'}")

        if event in ("ANNOTATION_CREATED", "ANNOTATION_UPDATED"):
            print("[fit] collecting single annotation")
            self._collect_annotation(data)

        elif event == "IMPORT_DONE":                     # bulk import finished
            for task in data.get("tasks", []):
                if task.get("annotations"):
                    self._collect_annotation(task["annotations"][0])  # first annotation per task

        # (OPTIONAL) manual "Start training" button in LS UI
        elif event == "START_TRAINING":
            print("[fit] START_TRAINING received – forcing retrain")
            self._maybe_retrain(force=True)

        # Add option to clear pool manually
        elif event == "CLEAR_POOL":
            print("[fit] CLEAR_POOL received – clearing AL pool")
            self.clear_pool()

        print("[fit] completed.\n")

    def _collect_annotation(self, payload):
        """
        Convert LS annotation(s) → persistent rows for active learning.
        Pool persists across instances via disk storage.
        """

        # List of payloads (batch from webhook)
        if isinstance(payload, list):
            for item in payload:
                self._collect_annotation(item)
            return

        # Skip if not labeled
        if "annotation" in payload and "task" in payload:
            ann  = payload["annotation"]
            task = payload["task"]
            text  = task.get("data", {}).get("text", "")
            spans = ann.get("result", [])

        # Raw annotation dict
        elif "task" in payload and "result" in payload:
            text  = payload["task"]["data"]["text"]
            spans = payload["result"]
        else:
            print("[collect] Malformed payload, skipping.")
            return

        if not text or not spans:
            print("[collect] Empty text or spans, skipping.")
            return

        # build LS-CSV-style label JSON
        span_objs = [
            {"start": s["value"]["start"],
            "end"  : s["value"]["end"],
            "labels": s["value"]["labels"]}
            for s in spans
        ]

        row = [text, json.dumps(span_objs, ensure_ascii=False)]

        # Load current pool from disk to check for duplicates
        self._load_pool()

        # Check for duplicates
        if not any(existing[0] == row[0] for existing in self._al_pool):
            self._al_pool.append(row)
            self._save_pool()  # Save to disk immediately
            print(f"[collect] Added new row: {row[0][:60]}...")
        else:
            print("[collect] Duplicate row, skipping.")

        print(f"[collect] pooled {len(self._al_pool)}/{self.AL_MIN_BATCH}")

        # Check if we have enough samples for automatic retraining
        if len(self._al_pool) >= self.AL_MIN_BATCH:
            print("[collect] AL pool full - starting automatic retrain")
            self._maybe_retrain()

    def _get_next_model_version(self) -> str:
        """Get the next model version number and manage version rotation"""
        # Find existing versions
        existing_versions = []
        for path in self.AL_MODEL_DIR.glob("v*"):
            if path.is_dir():
                try:
                    version_num = int(path.name[1:])  # Remove 'v' prefix
                    existing_versions.append(version_num)
                except ValueError:
                    continue

        # Determine next version
        if not existing_versions:
            next_version = 1
        else:
            next_version = max(existing_versions) + 1

        # Clean up old versions if we have too many
        if len(existing_versions) >= self.MAX_MODEL_VERSIONS:
            # Sort and remove oldest versions
            existing_versions.sort()
            versions_to_remove = existing_versions[:-self.MAX_MODEL_VERSIONS + 1]

            for version_to_remove in versions_to_remove:
                old_path = self.AL_MODEL_DIR / f"v{version_to_remove}"
                if old_path.exists():
                    print(f"[cleanup] Removing old model version: {old_path}")
                    shutil.rmtree(old_path)

        return f"v{next_version}"

    def _maybe_retrain(self, force: bool = False):
        """
        When enough rows collected (or force=True), retrain & hot-reload.
        """
        # Check if we have enough data to train
        if len(self._al_pool) < 3:
            print(f"[retrain] Not enough data to train (need ≥3 sentences, have {len(self._al_pool)})")
            if force:
                print("[retrain] Force training requested but insufficient data - skipping")
            return

        import csv, tempfile
        from label_studio_pos_backend.ls_csv_to_conllu import csv_to_conllu
        from src.pos_finetune import run

        print("[debug] Current _al_pool contents:")
        for i, row in enumerate(self._al_pool):
            print(f"  {i+1}. {row[0][:80]}")  # print sentence preview

        with tempfile.TemporaryDirectory() as td:
            # dump CSV
            csv_path = Path(td) / "al_pool.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["text", "label"])
                writer.writerows(self._al_pool)

            # print the csv
            print(f"[retrain] writing CSV to {csv_path} with {len(self._al_pool)} rows")

            # convert to CoNLL-U
            conllu_path = csv_path.with_suffix(".conllu")
            csv_to_conllu(csv_path, conllu_path)

            print("[debug] Dumping .conllu content:")
            with conllu_path.open("r", encoding="utf-8") as f:
                conllu_content = f.read()
                print(conllu_content)

            # Check if CoNLL-U file has content
            if not conllu_content.strip():
                print("[retrain] Empty CoNLL-U file generated - skipping training")
                return

            # Get next model version and create directory
            version_name = self._get_next_model_version()
            ckpt_dir = self.AL_MODEL_DIR / version_name
            ckpt_dir.mkdir(parents=True, exist_ok=True)

            print(f"[retrain] Training new model version: {version_name}")

            cfg = {
                "name": version_name,
                "model_type": "transformer",
                "conllu_path": str(conllu_path),
                "checkpoint_path": str(self.model_name_or_path),
                "pos_output_dir": str(ckpt_dir),
                "train_from_scratch": False,
                "search": False,
                "seed": 42,
                "max_length": 128,
            }

            try:
                metrics = run(cfg)
                print(f"[retrain] complete – F1={metrics['eval_f1']:.3f}")

                # hot-reload weights
                self._reload_model(ckpt_dir)

                # Clear the pool after successful training
                self.clear_pool()
                print("[retrain] AL pool cleared after successful training")

            except Exception as e:
                print(f"[retrain] Training failed: {e}")
                print("[retrain] Keeping AL pool for next attempt")
                # Clean up failed training directory
                if ckpt_dir.exists():
                    shutil.rmtree(ckpt_dir)

    def _reload_model(self, ckpt_dir: Path | str):
        """Swap tokenizer & model in RAM without restarting backend."""
        ckpt_dir = Path(ckpt_dir)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
            self.model     = AutoModelForTokenClassification.from_pretrained(ckpt_dir).eval()
            self.id2label  = self.model.config.id2label
            self.model_name_or_path = str(ckpt_dir)
            self.set("model_version", ckpt_dir.name)
            print(f"[reload] new checkpoint: {ckpt_dir}")
        except Exception as e:
            print(f"[reload] failed – keeping old weights. {e}")

# For compatibility, allow old name to still work
NewModel = HawaiianPOSModel
