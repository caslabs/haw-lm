from sklearn.feature_extraction.text import TfidfVectorizer
import pickle, pathlib, json, os, mlflow
import optuna
import numpy as np

def train_tfidf(corpus_file: str, cfg: dict):
    """
    Train a TF-IDF vectorizer using optional Optuna hyperparameter tuning.
    """

    # Load the corpus
    docs = [l.strip() for l in open(corpus_file, encoding="utf-8") if l.strip()]
    out_dir = pathlib.Path(cfg["checkpoint_path"]).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    def tfidf_objective(trial):
        # Suggest hyperparameters
        max_features = trial.suggest_categorical("max_features", [500, 1000, 3000, None])
        min_df       = trial.suggest_int("min_df", 1, 10)
        max_df       = trial.suggest_float("max_df", 0.5, 1.0)

        tfidf = TfidfVectorizer(
            lowercase=True,
            analyzer="word",
            token_pattern=r"\S+",
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
        ).fit(docs)

        X = tfidf.transform(docs)
        sparsity = 1.0 - X.nnz / (len(docs) * X.shape[1])
        return sparsity  # Lower is better (denser)

    if cfg.get("search", False):
        study = optuna.create_study(direction="minimize")
        study.optimize(tfidf_objective, n_trials=15)

        # Extract best params
        best_trial = study.best_trial
        best_params = best_trial.params

        tfidf = TfidfVectorizer(
            lowercase=True,
            analyzer="word",
            token_pattern=r"\S+",
            max_features=best_params["max_features"],
            min_df=best_params["min_df"],
            max_df=best_params["max_df"],
        ).fit(docs)

        # Save model
        trial_dir = out_dir / f"trial_{best_trial.number}"
        trial_dir.mkdir(exist_ok=True)
        pickle.dump(tfidf, open(trial_dir / "tfidf.pkl", "wb"))
        json.dump({tok: int(idx) for tok, idx in tfidf.vocabulary_.items()}, open(trial_dir / "vocab.json", "w"))

        X = tfidf.transform(docs)
        nnz = X.nnz
        dim = X.shape[1]

        mlflow.log_metric("tfidf_best_sparsity", 1.0 - nnz / (len(docs) * dim))
        mlflow.log_metric("tfidf_best_dim", dim)
        mlflow.log_metric("tfidf_best_nnz", nnz)
        mlflow.log_metric("tfidf_best_trial", best_trial.number)
        for k, v in best_params.items():
            mlflow.log_param(f"optuna_{k}", v)

        metrics = {
            "tfidf_best_sparsity": 1.0 - nnz / (len(docs) * dim),
            "tfidf_best_dim": dim,
            "tfidf_best_nnz": nnz,
            "tfidf_best_trial": best_trial.number,
            "tfidf_best_params": best_params,
            "tfidf_model_path": str(trial_dir),
        }

        return metrics, str(trial_dir)

    else:
        # Manual config fallback
        tfidf = TfidfVectorizer(
            lowercase=True,
            analyzer="word",
            token_pattern=r"\S+",
            max_features=int(cfg.get("tfidf_max_features", 0)) or None,
            min_df=int(cfg.get("tfidf_min_df", 1)),
            max_df=float(cfg.get("tfidf_max_df", 1.0)),
        ).fit(docs)

        pickle.dump(tfidf, open(out_dir / "tfidf.pkl", "wb"))
        json.dump({tok: int(idx) for tok, idx in tfidf.vocabulary_.items()}, open(out_dir / "vocab.json", "w"))

        X = tfidf.transform(docs)
        nnz = X.nnz
        dim = X.shape[1]
        metrics = {
            "tfidf_docs": len(docs),
            "tfidf_dim": dim,
            "tfidf_nnz": int(nnz),
            "tfidf_sparsity": float(1.0 - nnz / (len(docs) * dim)),
        }
        mlflow.log_metrics(metrics)
        return metrics, str(out_dir)
