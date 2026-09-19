from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import os, mlflow
from gensim.models.callbacks import CallbackAny2Vec
from tqdm import trange
from gensim.models.callbacks import CallbackAny2Vec
import optuna
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from pathlib import Path
import os
import mlflow
import time


class TQDMProgress(CallbackAny2Vec):
    """Show a TQDM progress bar for each epoch."""
    def __init__(self, epochs):
        self.pbar = trange(epochs, desc="Word2Vec Training")

    def on_epoch_end(self, model):
        self.pbar.update(1)

    def on_train_end(self, model):
        self.pbar.close()


class LossLogger(CallbackAny2Vec):
    """Log loss to MLflow & store last-epoch value for Discord."""
    def __init__(self, total_epochs):
        self.epoch      = 0
        self.total      = total_epochs
        self.loss_prev  = 0.0
        self.final_loss = None
        self.t0         = time.time()

    def on_epoch_end(self, model):
        # compute epoch-level loss delta
        cumulative = model.get_latest_training_loss()
        epoch_loss = cumulative - self.loss_prev
        self.loss_prev = cumulative

        mlflow.log_metric("w2v_epoch_loss",   epoch_loss, step=self.epoch)
        mlflow.log_metric("w2v_epoch",        self.epoch)

        self.epoch += 1
        if self.epoch == self.total:
            self.final_loss = epoch_loss
            mlflow.log_metric("w2v_final_loss", epoch_loss)
            mlflow.log_metric("w2v_train_sec",  time.time() - self.t0)


def train_word2vec(corpus_file: str, cfg: dict):
    """
    Train a Word2Vec model using Optuna for hyperparameter optimization.
    """


    def w2v_objective(trial):
        """
        Objective function for Optuna to optimize Word2Vec hyperparameters.
        """
        epochs      = trial.suggest_int("epochs", 5, 30)
        vector_size = trial.suggest_categorical("vector_size", [50, 100, 150, 200])
        window      = trial.suggest_int("window", 2, 10)
        min_count   = trial.suggest_int("min_count", 1, 5)
        sg          = trial.suggest_categorical("sg", [0, 1])
        negative    = trial.suggest_int("negative", 5, 15)
        seed        = cfg.get("seed", 42)

        sent_iter = LineSentence(corpus_file)
        loss_cb   = LossLogger(epochs)

        model = Word2Vec(
            sentences     = sent_iter,
            vector_size   = vector_size,
            window        = window,
            min_count     = min_count,
            sg            = sg,
            negative      = negative,
            epochs        = epochs,
            workers       = os.cpu_count(),
            seed          = seed,
            compute_loss  = True,
            callbacks     = [TQDMProgress(epochs), loss_cb],
        )

        # Train the model
        out_dir = Path(cfg["checkpoint_path"]) / f"trial_{trial.number}"
        out_dir.mkdir(parents=True, exist_ok=True)
        model.wv.save_word2vec_format(str(out_dir / "vectors.kv"), binary=True)
        model.save(str(out_dir / "word2vec.model"))
        return loss_cb.final_loss

    # If cfg['search'] is False, we skip the hyperparameter optimization
    if not cfg.get("search", True):
        # Default hyperparameters if not using Optuna
        epochs      = cfg.get("epochs", 10)
        vector_size = cfg.get("vector_size", 100)
        window      = cfg.get("window", 5)
        min_count   = cfg.get("min_count", 1)
        sg          = cfg.get("sg", 0)
        negative    = cfg.get("negative", 5)
        seed        = cfg.get("seed", 42)
        sent_iter   = LineSentence(corpus_file)
        loss_cb     = LossLogger(epochs)

        # Train the model with default parameters
        model = Word2Vec(
            sentences     = sent_iter,
            vector_size   = vector_size,
            window        = window,
            min_count     = min_count,
            sg            = sg,
            negative      = negative,
            epochs        = epochs,
            workers       = os.cpu_count(),
            seed          = seed,
            compute_loss  = True,
            callbacks     = [TQDMProgress(epochs), loss_cb],
        )

        # Log metrics
        mlflow.log_metric("w2v_final_loss", loss_cb.final_loss)
        mlflow.log_metric("w2v_train_sec", time.time() - loss_cb.t0)

        metrics = {
            "w2v_final_loss": loss_cb.final_loss,
            "w2v_train_sec": time.time() - loss_cb.t0,
            "w2v_model_path": str(Path(cfg["checkpoint_path"]) / "default_model"),
        }

        return metrics, str(Path(cfg["checkpoint_path"]) / "default_model")

    else:
        # Optimize with Optuna
        study = optuna.create_study(direction="minimize")
        study.optimize(w2v_objective, n_trials=20)

        # Best trial metrics
        best_trial = study.best_trial
        best_checkpoint = str(Path(cfg["checkpoint_path"]) / f"trial_{best_trial.number}")

        # Log clean metrics
        mlflow.log_metric("w2v_best_loss", float(best_trial.value))
        mlflow.log_metric("w2v_best_trial", best_trial.number)
        for k, v in best_trial.params.items():
            if isinstance(v, (int, float)):
                mlflow.log_param(f"optuna_{k}", v)
            else:
                mlflow.log_param(f"optuna_{k}", str(v))

        metrics = {
            "w2v_best_loss": best_trial.value,
            "w2v_best_trial": best_trial.number,
            "w2v_best_params": best_trial.params,
            "w2v_model_path": best_checkpoint,
        }

    return metrics, best_checkpoint
