"""Data source layer.

Two implementations behind one interface:

- ``DuckDBSource`` — local cache/dev store (``data/market.duckdb``). Always works.
- ``SnowflakeSource`` — system of record once seeded. Configured via env vars or
  Streamlit secrets: SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_PASSWORD (or
  SNOWFLAKE_PRIVATE_KEY_PATH), SNOWFLAKE_WAREHOUSE, SNOWFLAKE_DATABASE,
  SNOWFLAKE_SCHEMA.

Logical schema (identical in both backends):

    PRICES(date, ticker, open, high, low, close, adj_close, volume)
    FACTORS(date, factor, value)          -- long format, e.g. factor='MKT_RF'
    THEORIES(id, name, hypothesis, expression, config_json, metrics_json,
             verdict, created_at)
"""

from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "market.duckdb"

DDL = [
    """CREATE TABLE IF NOT EXISTS PRICES (
        date DATE, ticker VARCHAR, open DOUBLE, high DOUBLE, low DOUBLE,
        close DOUBLE, adj_close DOUBLE, volume DOUBLE
    )""",
    """CREATE TABLE IF NOT EXISTS FACTORS (
        date DATE, factor VARCHAR, value DOUBLE
    )""",
    """CREATE TABLE IF NOT EXISTS THEORIES (
        id VARCHAR, name VARCHAR, hypothesis VARCHAR, expression VARCHAR,
        config_json VARCHAR, metrics_json VARCHAR, verdict VARCHAR,
        created_at TIMESTAMP
    )""",
]


class DataSource:
    """Interface. Concrete classes implement _query/_execute/_insert_df."""

    name: str = "abstract"

    # ---- backend primitives -------------------------------------------------
    def _query(self, sql: str, params: Optional[list] = None) -> pd.DataFrame:
        raise NotImplementedError

    def _execute(self, sql: str, params: Optional[list] = None) -> None:
        raise NotImplementedError

    def _insert_df(self, table: str, df: pd.DataFrame) -> None:
        raise NotImplementedError

    def ensure_schema(self) -> None:
        for stmt in DDL:
            self._execute(stmt)

    # ---- prices --------------------------------------------------------------
    def available_tickers(self) -> list[str]:
        df = self._query("SELECT DISTINCT ticker FROM PRICES ORDER BY ticker")
        return df["ticker"].tolist() if not df.empty else []

    def coverage(self) -> pd.DataFrame:
        return self._query(
            """SELECT ticker, MIN(date) AS start, MAX(date) AS end,
                      COUNT(*) AS rows FROM PRICES GROUP BY ticker ORDER BY ticker"""
        )

    def get_price_panel(
        self,
        tickers: Optional[Sequence[str]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        field: str = "adj_close",
    ) -> pd.DataFrame:
        """Return a date x ticker panel of the requested field."""
        assert field in {"open", "high", "low", "close", "adj_close", "volume"}
        clauses, params = [], []
        if tickers:
            placeholders = ",".join(["?"] * len(tickers))
            clauses.append(f"ticker IN ({placeholders})")
            params.extend(list(tickers))
        if start:
            clauses.append("date >= ?")
            params.append(start)
        if end:
            clauses.append("date <= ?")
            params.append(end)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        df = self._query(
            f"SELECT date, ticker, {field} AS v FROM PRICES {where} ORDER BY date",
            params,
        )
        if df.empty:
            return pd.DataFrame()
        panel = df.pivot(index="date", columns="ticker", values="v")
        panel.index = pd.to_datetime(panel.index)
        return panel.sort_index()

    def write_prices(self, df: pd.DataFrame, replace_tickers: bool = True) -> int:
        """df: long format with the PRICES columns. Replaces existing rows for
        the same tickers by default so re-seeding is idempotent."""
        if df.empty:
            return 0
        if replace_tickers:
            tickers = df["ticker"].unique().tolist()
            placeholders = ",".join(["?"] * len(tickers))
            self._execute(f"DELETE FROM PRICES WHERE ticker IN ({placeholders})", tickers)
        self._insert_df("PRICES", df[["date", "ticker", "open", "high", "low",
                                      "close", "adj_close", "volume"]])
        return len(df)

    # ---- factors ---------------------------------------------------------------
    def get_factors(
        self,
        names: Optional[Sequence[str]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Return date x factor panel of daily factor returns (decimal)."""
        clauses, params = [], []
        if names:
            placeholders = ",".join(["?"] * len(names))
            clauses.append(f"factor IN ({placeholders})")
            params.extend(list(names))
        if start:
            clauses.append("date >= ?")
            params.append(start)
        if end:
            clauses.append("date <= ?")
            params.append(end)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        df = self._query(f"SELECT date, factor, value FROM FACTORS {where}", params)
        if df.empty:
            return pd.DataFrame()
        panel = df.pivot(index="date", columns="factor", values="value")
        panel.index = pd.to_datetime(panel.index)
        return panel.sort_index()

    def available_factors(self) -> list[str]:
        df = self._query("SELECT DISTINCT factor FROM FACTORS ORDER BY factor")
        return df["factor"].tolist() if not df.empty else []

    def write_factors(self, df: pd.DataFrame) -> int:
        """df: long format (date, factor, value). Replaces same factor names."""
        if df.empty:
            return 0
        names = df["factor"].unique().tolist()
        placeholders = ",".join(["?"] * len(names))
        self._execute(f"DELETE FROM FACTORS WHERE factor IN ({placeholders})", names)
        self._insert_df("FACTORS", df[["date", "factor", "value"]])
        return len(df)

    # ---- theories ----------------------------------------------------------------
    def save_theory(
        self,
        name: str,
        hypothesis: str,
        expression: str,
        config: dict,
        metrics: dict,
        verdict: str = "untested",
    ) -> str:
        tid = uuid.uuid4().hex[:12]
        self._execute(
            "INSERT INTO THEORIES VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [tid, name, hypothesis, expression, json.dumps(config),
             json.dumps(metrics, default=float), verdict,
             datetime.now(timezone.utc).replace(tzinfo=None)],
        )
        return tid

    def list_theories(self) -> pd.DataFrame:
        return self._query("SELECT * FROM THEORIES ORDER BY created_at DESC")

    def delete_theory(self, tid: str) -> None:
        self._execute("DELETE FROM THEORIES WHERE id = ?", [tid])


class DuckDBSource(DataSource):
    """Local store. Connections are opened per operation (with a brief retry on
    file-lock contention) so the app, seed scripts, and tests can coexist."""

    name = "duckdb"

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path or DEFAULT_DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.ensure_schema()

    @contextmanager
    def _connect(self):
        import duckdb

        last_err = None
        for _ in range(40):
            try:
                con = duckdb.connect(str(self.db_path))
                break
            except duckdb.IOException as e:
                last_err = e
                time.sleep(0.25)
        else:
            raise last_err
        try:
            yield con
        finally:
            con.close()

    def ensure_schema(self) -> None:
        with self._connect() as con:
            for stmt in DDL:
                con.execute(stmt)

    def _query(self, sql, params=None):
        with self._connect() as con:
            return con.execute(sql, params or []).df()

    def _execute(self, sql, params=None):
        with self._connect() as con:
            con.execute(sql, params or [])

    def _insert_df(self, table, df):
        with self._connect() as con:
            con.register("_tmp_df", df)
            con.execute(f"INSERT INTO {table} SELECT * FROM _tmp_df")
            con.unregister("_tmp_df")


@dataclass
class SnowflakeConfig:
    account: str
    user: str
    warehouse: str
    database: str
    schema: str = "PUBLIC"
    password: Optional[str] = None
    private_key_path: Optional[str] = None
    role: Optional[str] = None

    @classmethod
    def from_env(cls) -> Optional["SnowflakeConfig"]:
        def get(key: str) -> Optional[str]:
            val = os.environ.get(key)
            if val:
                return val
            try:  # Streamlit secrets, if running inside the app
                import streamlit as st

                return st.secrets.get(key)  # type: ignore[attr-defined]
            except Exception:
                return None

        account, user = get("SNOWFLAKE_ACCOUNT"), get("SNOWFLAKE_USER")
        if not (account and user):
            return None
        return cls(
            account=account,
            user=user,
            warehouse=get("SNOWFLAKE_WAREHOUSE") or "COMPUTE_WH",
            database=get("SNOWFLAKE_DATABASE") or "QUANT",
            schema=get("SNOWFLAKE_SCHEMA") or "PUBLIC",
            password=get("SNOWFLAKE_PASSWORD"),
            private_key_path=get("SNOWFLAKE_PRIVATE_KEY_PATH"),
            role=get("SNOWFLAKE_ROLE"),
        )


class SnowflakeSource(DataSource):
    name = "snowflake"

    def __init__(self, config: SnowflakeConfig):
        try:
            import snowflake.connector  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "snowflake-connector-python is not installed. "
                "Run: pip install snowflake-connector-python"
            ) from e
        self.config = config
        self._con = self._connect()
        self.ensure_schema()

    def _connect(self):
        import snowflake.connector

        kwargs = dict(
            account=self.config.account,
            user=self.config.user,
            warehouse=self.config.warehouse,
            database=self.config.database,
            schema=self.config.schema,
        )
        if self.config.role:
            kwargs["role"] = self.config.role
        if self.config.private_key_path:
            from cryptography.hazmat.primitives import serialization

            key_bytes = Path(self.config.private_key_path).read_bytes()
            pkey = serialization.load_pem_private_key(key_bytes, password=None)
            kwargs["private_key"] = pkey.private_bytes(
                encoding=serialization.Encoding.DER,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )
        else:
            kwargs["password"] = self.config.password
        return snowflake.connector.connect(**kwargs)

    @staticmethod
    def _qmark(sql: str) -> str:
        return sql  # snowflake connector uses qmark paramstyle when configured

    def _query(self, sql, params=None):
        cur = self._con.cursor()
        try:
            cur.execute(sql.replace("?", "%s"), params or [])
            df = cur.fetch_pandas_all()
            df.columns = [c.lower() for c in df.columns]
            return df
        finally:
            cur.close()

    def _execute(self, sql, params=None):
        cur = self._con.cursor()
        try:
            cur.execute(sql.replace("?", "%s"), params or [])
        finally:
            cur.close()

    def _insert_df(self, table, df):
        from snowflake.connector.pandas_tools import write_pandas

        out = df.copy()
        out.columns = [c.upper() for c in out.columns]
        write_pandas(self._con, out, table.upper())


def get_source(prefer: str = "auto") -> DataSource:
    """Resolve the active data source.

    prefer='auto' uses Snowflake when credentials are configured, else DuckDB.
    """
    if prefer in ("auto", "snowflake"):
        cfg = SnowflakeConfig.from_env()
        if cfg is not None:
            try:
                return SnowflakeSource(cfg)
            except Exception:
                if prefer == "snowflake":
                    raise
        elif prefer == "snowflake":
            raise RuntimeError(
                "Snowflake requested but SNOWFLAKE_ACCOUNT / SNOWFLAKE_USER not set."
            )
    return DuckDBSource()
