"""Bridge to the firm-internal ``snowflake_utilities`` package.

On machines where the internal package is installed, ``read_sql`` delegates to
it directly. Elsewhere, it falls back to ``snowflake-connector-python`` with
SNOWFLAKE_* env vars so the same code paths remain runnable. If neither is
available the caller gets a clear ImportError (and ``get_source`` will have
already fallen back to the local DuckDB store).
"""

from __future__ import annotations

import os
from typing import Optional

import pandas as pd

try:  # firm-internal package, present on work machines
    import snowflake_utilities as _firm  # type: ignore

    HAVE_FIRM_UTILS = True
except ImportError:
    _firm = None
    HAVE_FIRM_UTILS = False


def is_available() -> bool:
    """True if any Snowflake execution path exists."""
    if HAVE_FIRM_UTILS:
        return True
    return bool(os.environ.get("SNOWFLAKE_ACCOUNT") and os.environ.get("SNOWFLAKE_USER"))


def read_sql(
    query: str,
    return_df: bool = True,
    warehouse: Optional[str] = None,
    database: Optional[str] = None,
    schema: Optional[str] = None,
) -> pd.DataFrame:
    if HAVE_FIRM_UTILS:
        return _firm.read_sql(
            query=query,
            return_df=return_df,
            warehouse=warehouse,
            database=database,
            schema=schema,
        )
    return _fallback_read_sql(query, warehouse=warehouse, database=database,
                              schema=schema)


def _fallback_read_sql(query, warehouse=None, database=None, schema=None):
    from quantdash.data.source import SnowflakeConfig

    cfg = SnowflakeConfig.from_env()
    if cfg is None:
        raise ImportError(
            "snowflake_utilities is not installed and SNOWFLAKE_ACCOUNT / "
            "SNOWFLAKE_USER are not set — no Snowflake path available."
        )
    import snowflake.connector

    con = snowflake.connector.connect(
        account=cfg.account,
        user=cfg.user,
        password=cfg.password,
        warehouse=warehouse or cfg.warehouse,
        database=database or cfg.database,
        schema=schema or cfg.schema,
    )
    try:
        cur = con.cursor()
        cur.execute(query)
        return cur.fetch_pandas_all()
    finally:
        con.close()
