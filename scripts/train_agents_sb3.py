from src.rl.train_sb3 import train_all_pairs_sb3
from src.preflight import run_checks

if __name__ == "__main__":
    run_checks()
    train_all_pairs_sb3()
