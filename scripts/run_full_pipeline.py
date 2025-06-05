"""Convenience script to run the full research pipeline."""

import logging

from src.data.downloader import batch as dl
from src.data.feature_engineer import batch as fe
from src.autoencoder.train_cae import train_cae
from src.clustering.cluster_utils import cluster_latents
from src.clustering.select_pairs import select_pairs
from src.rl.train_agent import train_all_pairs
from src.backtest.backtester import run_backtests

logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

STEPS = [
    ("download data", dl),
    ("feature engineer", fe),
    ("train CAE", train_cae),
    ("cluster latents", cluster_latents),
    ("select pairs", select_pairs),
    ("train agents", train_all_pairs),
    ("run backtests", run_backtests),
]

def main() -> None:
    """Run each stage of the pipeline sequentially.

    Errors in a stage are logged but do not abort the remaining steps.
    """

    for name, func in STEPS:
        try:
            logger.info("Starting %s", name)
            func()
        except Exception as exc:  # noqa: BLE001
            logger.error("Stage %s failed: %s", name, exc)

if __name__ == '__main__':
    main()
