
from src.data.downloader import batch as dl
from src.data.feature_engineer import batch as fe
from src.autoencoder.train_cae import train_cae
from src.clustering.cluster_utils import cluster_latents
from src.clustering.select_pairs import select_pairs
from src.rl.train_agent import train_all_pairs
from src.backtest.backtester import run_backtests

def main():
    dl(); fe(); train_cae()
    cluster_latents(); select_pairs()
    train_all_pairs(); run_backtests()

if __name__ == '__main__':
    main()
