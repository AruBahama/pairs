"""Bayesian hyperparameter search for the CAE and clustering steps."""

import optuna
import tensorflow as tf
from sklearn.metrics import calinski_harabasz_score
from src.autoencoder.train_cae import train_cae
from src.clustering.cluster_utils import cluster_latents
from src.config import LOG_DIR


def objective(trial: optuna.Trial) -> float:
    window_length = trial.suggest_int('window_length', 20, 120)
    latent_dim = trial.suggest_int('latent_dim', 5, 30)
    n_clusters = trial.suggest_int('n_clusters', 5, 20)

    try:
        _, ticker_latent, _ = train_cae(
            window_length=window_length,
            latent_dim=latent_dim,
            save=False,
            callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
        )
        _, labels = cluster_latents(ticker_latent, n_clusters=n_clusters, save=False)
        score = calinski_harabasz_score(ticker_latent, labels)
        return -float(score)
    except Exception as exc:
        print(f"Trial failed: {exc}")
        return float("inf")


def main() -> None:
    study = optuna.create_study()
    study.optimize(objective, n_trials=10)
    study.trials_dataframe().to_csv(LOG_DIR/"optuna_trials.csv", index=False)
    print("Best parameters:", study.best_params)


if __name__ == "__main__":
    main()
