import pandas as pd
import sqlite3
from pathlib import Path
import typer
from loguru import logger

from proj_data_mining_northwind import config

app = typer.Typer()


def get_rfm_features(
    db_path: Path = config.PROCESSED_DATA_DIR / "northwind.db",
) -> pd.DataFrame:
    """Calculate RFM features from the Northwind database."""
    logger.info(f"Connecting to database at: {db_path}")
    conn = sqlite3.connect(db_path)

    # Read tables into pandas DataFrames
    orders = pd.read_sql_query("SELECT * FROM orders", conn)
    order_details = pd.read_sql_query("SELECT * FROM order_details", conn)

    # Convert order_date to datetime
    orders["order_date"] = pd.to_datetime(orders["order_date"])

    # Calculate Monetary Value
    order_details["monetary"] = (
        order_details["unit_price"] * order_details["quantity"] * (1 - order_details["discount"])
    )
    monetary = order_details.groupby("order_id")["monetary"].sum().reset_index()

    # Merge with orders table
    orders = pd.merge(orders, monetary, on="order_id")

    # Calculate Recency, Frequency, and Monetary values for each customer
    snapshot_date = orders["order_date"].max() + pd.DateOffset(days=1)
    rfm = orders.groupby("customer_id").agg(
        {
            "order_date": lambda x: (snapshot_date - x.max()).days,
            "order_id": "count",
            "monetary": "sum",
        }
    )

    # Rename columns
    rfm.rename(
        columns={
            "order_date": "recency",
            "order_id": "frequency",
            "monetary": "monetary",
        },
        inplace=True,
    )

    conn.close()
    return rfm


@app.command()
def create_features(
    output_path: Path = typer.Option(
        config.PROCESSED_DATA_DIR / "rfm_features.parquet",
        help="Path to save the RFM features.",
    ),
):
    """Create and save RFM features."""
    logger.info("Creating RFM features...")
    rfm_features = get_rfm_features()
    logger.info(f"Saving features to: {output_path}")
    rfm_features.to_parquet(output_path)


if __name__ == "__main__":
    app()