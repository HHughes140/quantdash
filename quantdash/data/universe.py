"""Ticker universes.

- INSURANCE_UNIVERSE: the core universe — US-listed insurance complex by
  subsector (P&C, specialty/E&S, life, reinsurance, brokers, health,
  title/mortgage, insurtech).
- DEFAULT_UNIVERSE: broad large-cap universe kept for cross-sector context.
- BENCHMARKS: SPY (market), KIE (S&P Insurance Select), IAK (US Insurance).
"""

INSURANCE_UNIVERSE = {
    "P&C": [
        "PGR", "ALL", "TRV", "CB", "AIG", "HIG", "CINF", "WRB", "ACGL", "MKL",
        "AFG", "ORI", "CNA", "THG", "KMPR", "MCY", "SIGI", "ERIE",
    ],
    "Specialty & E&S": [
        "KNSL", "RLI", "PLMR", "SKWD", "HCI", "UVE", "AMSF", "AGO",
    ],
    "Life & Retirement": [
        "MET", "PRU", "AFL", "UNM", "GL", "LNC", "PFG", "EQH", "CRBG", "BHF",
        "VOYA", "JXN",
    ],
    "Reinsurance": [
        "RGA", "RNR", "EG", "SPNT", "HG",
    ],
    "Brokers & Services": [
        "MMC", "AON", "AJG", "WTW", "BRO", "RYAN", "BWIN", "GSHD", "CRVL",
    ],
    "Health": [
        "UNH", "ELV", "CI", "HUM", "CNC", "MOH", "OSCR",
    ],
    "Title & Mortgage": [
        "FNF", "FAF", "STC", "RDN", "MTG", "ESNT", "NMIH",
    ],
    "Insurtech": [
        "LMND", "ROOT", "HIPO", "EVER", "GOCO",
    ],
}

INSURANCE_TICKERS = [t for ts in INSURANCE_UNIVERSE.values() for t in ts]

# ticker -> subsector
SUBSECTOR = {t: sub for sub, ts in INSURANCE_UNIVERSE.items() for t in ts}

BENCHMARKS = ["SPY", "KIE", "IAK"]

DEFAULT_UNIVERSE = [
    # Tech / communication
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AVGO", "ORCL", "CRM", "ADBE",
    "AMD", "INTC", "QCOM", "TXN", "MU", "AMAT", "LRCX", "KLAC", "NOW", "INTU",
    "IBM", "CSCO", "ACN", "PANW", "SNPS", "CDNS", "NFLX", "DIS", "CMCSA",
    "TMUS", "VZ", "T",
    # Consumer discretionary
    "AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "TJX", "BKNG", "CMG",
    "MAR", "GM", "F", "ROST", "YUM",
    # Consumer staples
    "PG", "KO", "PEP", "COST", "WMT", "PM", "MO", "MDLZ", "CL", "KMB", "GIS",
    "KHC", "STZ", "TGT",
    # Health care
    "UNH", "JNJ", "LLY", "ABBV", "MRK", "PFE", "TMO", "ABT", "DHR", "BMY",
    "AMGN", "GILD", "ISRG", "CVS", "MDT", "VRTX", "REGN", "CI", "HUM", "SYK",
    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP", "SPGI", "V",
    "MA", "PYPL", "COF", "MET", "AIG", "CB", "PGR", "TRV", "ALL", "USB", "PNC",
    # Industrials
    "CAT", "DE", "UNP", "UPS", "FDX", "HON", "GE", "MMM", "BA", "LMT", "RTX",
    "NOC", "GD", "EMR", "ETN", "ITW", "CSX", "NSC", "WM",
    # Energy / materials / utilities / REITs
    "XOM", "CVX", "COP", "SLB", "EOG", "PSX", "MPC", "OXY", "LIN", "APD",
    "SHW", "FCX", "NEM", "NUE", "DOW", "NEE", "DUK", "SO", "D", "AEP", "EXC",
    "AMT", "PLD", "CCI", "EQIX", "SPG", "O",
]
