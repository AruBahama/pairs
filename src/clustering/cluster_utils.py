
import numpy as np
import joblib
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from ..config import N_CLUSTERS, LOG_DIR

# Default location of the latent vectors produced by the autoencoder
path = LOG_DIR / "ticker_latent.npy"

def cluster_latents(
    Z: np.ndarray | None = None, n_clusters: int = N_CLUSTERS, save: bool = True
):
    """Cluster latent vectors and optionally save the labels."""

    save_output = Z is None
    scaler_path = LOG_DIR / "cluster_scaler.joblib"
    if Z is None:
        if not path.exists():
            raise FileNotFoundError(f"{path} missing – run train_cae() first")
        Z = np.load(path)
        scaler = StandardScaler()
        Zs = scaler.fit_transform(Z)
        joblib.dump(scaler, scaler_path)
    else:
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
            Zs = scaler.transform(Z)
        else:
            scaler = StandardScaler()
            Zs = scaler.fit_transform(Z)

    clust = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    labels = clust.fit_predict(Zs)

    if save and save_output:
        np.save(LOG_DIR / "labels.npy", labels)

    return Zs, labels
