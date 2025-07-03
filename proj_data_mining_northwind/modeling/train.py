import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
import joblib
from pathlib import Path
import typer
from loguru import logger

from proj_data_mining_northwind import config

app = typer.Typer()


def train_models(
    features_path: Path = config.PROCESSED_DATA_DIR / "rfm_features.parquet",
    models_dir: Path = config.MODELS_DIR,
) -> pd.DataFrame:
    """Train clustering models and add cluster labels to the features."""
    logger.info(f"Loading features from: {features_path}")
    rfm_features = pd.read_parquet(features_path)

    # Scale the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(rfm_features)

    joblib.dump(scaler, models_dir / "scaler.joblib")

    # Train K-Means model
    logger.info("Training K-Means model...")
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    rfm_features["kmeans_cluster"] = kmeans.fit_predict(scaled_features)
    joblib.dump(kmeans, models_dir / "kmeans_model.joblib")

    # Train Hierarchical Clustering model
    logger.info("Training Hierarchical Clustering model...")
    agg_clustering = AgglomerativeClustering(n_clusters=3)
    rfm_features["agg_cluster"] = agg_clustering.fit_predict(scaled_features)
    joblib.dump(agg_clustering, models_dir / "agg_clustering_model.joblib")

    # Train DBSCAN model
    logger.info("Training DBSCAN model...")
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    rfm_features["dbscan_cluster"] = dbscan.fit_predict(scaled_features)
    joblib.dump(dbscan, models_dir / "dbscan_model.joblib")

    return rfm_features


@app.command()
def train(
    output_path: Path = typer.Option(
        config.PROCESSED_DATA_DIR / "labeled_features.parquet",
        help="Path to save the features with cluster labels.",
    ),
):
    """Train clustering models and save the labeled features."""
    logger.info("Training clustering models...")
    labeled_features = train_models()
    logger.info(f"Saving labeled features to: {output_path}")
    labeled_features.to_parquet(output_path)


if __name__ == "__main__":
    app()