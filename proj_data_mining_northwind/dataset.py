

import sqlite3
import re
from pathlib import Path
import typer
from loguru import logger

from proj_data_mining_northwind import config

app = typer.Typer()

def parse_sql_dump(sql_content: str) -> tuple[list[str], list[str]]:
    """Parse the SQL dump and separate CREATE and INSERT statements."""
    # Remove comments and clean up the SQL content
    sql_content = re.sub(r"--.*\n", "", sql_content)
    sql_content = re.sub(r"\/\*.*?\*\/", "", sql_content, flags=re.DOTALL)
    sql_content = re.sub(r"SET .*\n", "", sql_content)
    sql_content = re.sub(r"SELECT pg_catalog.set_config\('search_path', '', false\);", "", sql_content)
    sql_content = re.sub(r"character varying\((\d+)\)", r"VARCHAR(\1)", sql_content)
    sql_content = sql_content.replace("bytea", "BLOB")
    sql_content = re.sub(r"\\x", "NULL", sql_content)

    statements = [s.strip() for s in sql_content.split(";") if s.strip()]

    create_statements = [s for s in statements if s.startswith("CREATE TABLE")]
    insert_statements = [s for s in statements if s.startswith("INSERT INTO")]

    return create_statements, insert_statements

def execute_sql_statements(db_path: Path, statements: list[str]):
    """Execute a list of SQL statements on a SQLite database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    for statement in statements:
        try:
            cursor.execute(statement)
            conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Error executing statement: {statement}\n{e}")
    conn.close()

@app.command()
def create_database(
    sql_file_path: Path = typer.Option(config.EXTERNAL_DATA_DIR / "northwind.sql", help="Path to the SQL script."),
    db_path: Path = typer.Option(config.PROCESSED_DATA_DIR / "northwind.db", help="Path to the SQLite database."),
):
    """Create a SQLite database from an SQL script."""
    logger.info(f"Reading SQL script from: {sql_file_path}")
    with open(sql_file_path, "r") as f:
        sql_content = f.read()

    create_statements, insert_statements = parse_sql_dump(sql_content)

    # Drop tables before creating them
    drop_statements = ["DROP TABLE IF EXISTS " + re.search(r"CREATE TABLE (\w+)", s).group(1) for s in create_statements]

    logger.info(f"Creating database at: {db_path}")
    execute_sql_statements(db_path, drop_statements + create_statements + insert_statements)

if __name__ == "__main__":
    app()

