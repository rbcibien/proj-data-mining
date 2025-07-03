import pandas as pd
import joblib
from pathlib import Path
import typer
from loguru import logger

from proj_data_mining_northwind import config

app = typer.Typer()


def predict_cluster(
    recency: int,
    frequency: int,
    monetary: float,
    model_path: Path = config.MODELS_DIR / "kmeans_model.joblib",
    scaler_path: Path = config.MODELS_DIR / "scaler.joblib",
) -> int:
    """Predict the cluster for a new customer."""
    logger.info(f"Loading model from: {model_path}")
    model = joblib.load(model_path)
    logger.info(f"Loading scaler from: {scaler_path}")
    scaler = joblib.load(scaler_path)

    # Create a DataFrame from the input data
    new_data = pd.DataFrame(
        [{"recency": recency, "frequency": frequency, "monetary": monetary}]
    )

    # Scale the new data
    scaled_data = scaler.transform(new_data)

    # Predict the cluster
    prediction = model.predict(scaled_data)

    return prediction[0]


@app.command()
def predict(
    recency: int = typer.Option(..., help="Recency value."),
    frequency: int = typer.Option(..., help="Frequency value."),
    monetary: float = typer.Option(..., help="Monetary value."),
):
    """Predict the cluster for a new customer."""
    cluster = predict_cluster(recency, frequency, monetary)
    logger.info(f"The predicted cluster is: {cluster}")


if __name__ == "__main__":
    app()