"""Read-only Snowflake data source using Axioma tables via snowflake_utilities.

This source is intentionally read-only.

Price/volume feed:
    AXIOMA.FUNDAMENTAL.STOCKS
    (adj_close is a synthetic total-return index built by cumulating
    _1_DAY_RETURN; volume proxied by _20_DAY_ADV)

Factor returns:
    Loaded through quantdash.engine.factors.load_axioma_factor_returns_from_snowflake

Theories cannot be written here — the app persists them to the local DuckDB
store instead (see get_theory_store in quantdash.data).
"""

from __future__ import annotations

from typing import Optional, Sequence

import pandas as pd

from quantdash.engine.factors import (
    AXIOMA_FACTOR_NAMES,
    load_axioma_factor_returns_from_snowflake,
)
from quantdash.engine.snowflake_utilities import read_sql
from .source import DataSource
from .universe import DEFAULT_UNIVERSE, INSURANCE_TICKERS

PRICE_TABLE = "AXIOMA.FUNDAMENTAL.STOCKS"

RISK_MODEL = "WW4"
HORIZON = "SH"

# Universe visible through this source: insurance complex + broad large caps
UNIVERSE_TICKERS = sorted(set(INSURANCE_TICKERS) | set(DEFAULT_UNIVERSE))


class SnowflakeUtilitiesSource(DataSource):
    name = "snowflake_axioma_read_only"
    read_only = True

    @staticmethod
    def _sql_literal(value) -> str:
        if value is None:
            return "NULL"
        if isinstance(value, pd.Timestamp):
            return "'" + value.strftime("%Y-%m-%d") + "'"
        if hasattr(value, "strftime"):
            return "'" + value.strftime("%Y-%m-%d") + "'"
        if isinstance(value, str):
            return "'" + value.replace("'", "''") + "'"
        return str(value)

    @classmethod
    def _sql_list(cls, values: Sequence[str]) -> str:
        cleaned = [
            str(v).strip().upper().replace("'", "''")
            for v in values
            if v is not None and str(v).strip()
        ]
        if not cleaned:
            return "''"
        return ", ".join(f"'{v}'" for v in sorted(set(cleaned)))

    @staticmethod
    def _read(sql: str) -> pd.DataFrame:
        df = read_sql(
            query=sql,
            return_df=True,
            warehouse="WHSE_TEAM_WILHELM_001",
            database="AXIOMA",
            schema="FUNDAMENTAL",
        )
        if df is None:
            return pd.DataFrame()
        df.columns = [str(c).lower() for c in df.columns]
        return df

    # ---- reads -------------------------------------------------------------
    def available_tickers(self) -> list[str]:
        ticker_sql = self._sql_list(UNIVERSE_TICKERS)
        sql = f"""
        SELECT DISTINCT
            TICKER
        FROM {PRICE_TABLE}
        WHERE RISK_MODEL = '{RISK_MODEL}'
          AND HORIZON = '{HORIZON}'
          AND TICKER IN ({ticker_sql})
          AND TICKER IS NOT NULL
          AND _1_DAY_RETURN IS NOT NULL
        ORDER BY TICKER
        """
        df = self._read(sql)
        if df.empty or "ticker" not in df.columns:
            return []
        return df["ticker"].dropna().astype(str).sort_values().tolist()

    def coverage(self) -> pd.DataFrame:
        ticker_sql = self._sql_list(UNIVERSE_TICKERS)
        sql = f"""
        SELECT
            TICKER AS ticker,
            MIN(DDATE) AS start_date,
            MAX(DDATE) AS end_date,
            COUNT(*) AS row_count
        FROM {PRICE_TABLE}
        WHERE RISK_MODEL = '{RISK_MODEL}'
          AND HORIZON = '{HORIZON}'
          AND TICKER IN ({ticker_sql})
          AND TICKER IS NOT NULL
          AND _1_DAY_RETURN IS NOT NULL
        GROUP BY TICKER
        ORDER BY TICKER
        """
        df = self._read(sql)
        if df.empty:
            return pd.DataFrame(columns=["ticker", "start", "end", "rows"])
        df = df.rename(columns={"start_date": "start", "end_date": "end",
                                "row_count": "rows"})
        df["start"] = pd.to_datetime(df["start"], errors="coerce")
        df["end"] = pd.to_datetime(df["end"], errors="coerce")
        return df

    def get_price_panel(
        self,
        tickers: Optional[Sequence[str]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        field: str = "adj_close",
    ) -> pd.DataFrame:
        price_like_fields = {"adj_close", "close", "open", "high", "low"}
        field_map = {
            "adj_close": "_1_DAY_RETURN",
            "close": "_1_DAY_RETURN",
            "open": "_1_DAY_RETURN",
            "high": "_1_DAY_RETURN",
            "low": "_1_DAY_RETURN",
            "volume": "_20_DAY_ADV",
        }
        if field not in field_map:
            raise ValueError(f"Unsupported price field: {field}")
        value_col = field_map[field]

        clauses = [
            f"RISK_MODEL = '{RISK_MODEL}'",
            f"HORIZON = '{HORIZON}'",
            "TICKER IS NOT NULL",
            f"{value_col} IS NOT NULL",
        ]
        if tickers:
            clauses.append(f"TICKER IN ({self._sql_list(tickers)})")
        else:
            clauses.append(f"TICKER IN ({self._sql_list(UNIVERSE_TICKERS)})")
        if start:
            clauses.append(f"DDATE >= {self._sql_literal(start)}")
        if end:
            clauses.append(f"DDATE <= {self._sql_literal(end)}")
        where = " AND ".join(clauses)

        sql = f"""
        SELECT
            DDATE AS date,
            TICKER AS ticker,
            TRY_TO_DOUBLE({value_col}) AS v
        FROM {PRICE_TABLE}
        WHERE {where}
        ORDER BY DDATE, TICKER
        """
        df = self._read(sql)
        if df.empty:
            return pd.DataFrame()

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["v"] = pd.to_numeric(df["v"], errors="coerce")
        if field in price_like_fields:
            df["v"] = df["v"] / 100.0
        df = df.dropna(subset=["date", "ticker", "v"])
        if df.empty:
            return pd.DataFrame()

        panel = df.pivot_table(index="date", columns="ticker", values="v",
                               aggfunc="last").sort_index()
        if field in price_like_fields:
            # cumulate daily returns into a synthetic total-return price index
            panel = panel.clip(lower=-0.9999)
            panel = (1.0 + panel.fillna(0.0)).cumprod()
        return panel

    def get_factors(
        self,
        names: Optional[Sequence[str]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        factors = load_axioma_factor_returns_from_snowflake(
            history_days=3650,
            factor_names=list(names) if names else AXIOMA_FACTOR_NAMES,
        )
        if factors.empty:
            return pd.DataFrame()
        factors = factors.copy()
        factors.index = pd.to_datetime(factors.index, errors="coerce")
        factors = factors[~factors.index.isna()].sort_index()
        if start:
            factors = factors.loc[factors.index >= pd.Timestamp(start)]
        if end:
            factors = factors.loc[factors.index <= pd.Timestamp(end)]
        return factors

    def available_factors(self) -> list[str]:
        factors = load_axioma_factor_returns_from_snowflake(
            history_days=30,
            factor_names=AXIOMA_FACTOR_NAMES,
        )
        if factors.empty:
            return []
        return sorted(list(factors.columns))

    # ---- writes: disabled ----------------------------------------------------
    def write_prices(self, *args, **kwargs) -> int:
        raise PermissionError("Read-only Snowflake source: write_prices is disabled.")

    def write_factors(self, *args, **kwargs) -> int:
        raise PermissionError("Read-only Snowflake source: write_factors is disabled.")

    def save_theory(self, *args, **kwargs) -> str:
        raise PermissionError("Read-only Snowflake source: save_theory is disabled.")

    def list_theories(self) -> pd.DataFrame:
        return pd.DataFrame()

    def delete_theory(self, *args, **kwargs) -> None:
        raise PermissionError("Read-only Snowflake source: delete_theory is disabled.")
