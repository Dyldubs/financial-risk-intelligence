"""
Build the SQLite analytics database from synthetic transaction data.

Creates three tables:
  - transactions  : core transaction records with account and merchant FKs
  - accounts      : account-level metadata (type, open date, credit limit)
  - merchants     : merchant reference data (name, category, risk tier)

Usage:
    python scripts/build_db.py

Output:
    data/transactions.db  (SQLite, ~2 MB for 10,000 rows)
"""

import sys
import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_transactions
from src.config import DATA_DIR

DB_PATH = DATA_DIR / "transactions.db"

# --------------------------------------------------------------------------- #
# Reference data
# --------------------------------------------------------------------------- #

MERCHANT_CATALOGUE = [
    # (name, category, risk_tier)
    # Tier 0 — low risk
    ("Woolworths",          "Grocery",          0),
    ("Coles",               "Grocery",          0),
    ("ALDI Australia",      "Grocery",          0),
    ("Chemist Warehouse",   "Pharmacy",         0),
    ("Priceline Pharmacy",  "Pharmacy",         0),
    ("Kmart",               "Retail",           0),
    ("Target Australia",    "Retail",           0),
    ("Big W",               "Retail",           0),
    ("Bunnings Warehouse",  "Home Improvement", 0),
    ("IKEA Australia",      "Furniture",        0),
    ("Officeworks",         "Office Supplies",  0),
    ("JB Hi-Fi",            "Electronics",      0),
    ("Harvey Norman",       "Electronics",      0),
    ("Dan Murphy's",        "Liquor",           0),
    ("BWS",                 "Liquor",           0),
    ("Ampol",               "Fuel",             0),
    ("BP Australia",        "Fuel",             0),
    ("Shell",               "Fuel",             0),
    ("McDonald's",          "Fast Food",        0),
    ("KFC Australia",       "Fast Food",        0),
    ("Subway",              "Fast Food",        0),
    ("Uber Eats",           "Food Delivery",    0),
    ("DoorDash AU",         "Food Delivery",    0),
    ("Menulog",             "Food Delivery",    0),
    ("Netflix AU",          "Streaming",        0),
    ("Spotify AU",          "Streaming",        0),
    ("Qantas",              "Travel",           0),
    ("Virgin Australia",    "Travel",           0),
    ("Airbnb AU",           "Accommodation",    0),
    ("Booking.com AU",      "Accommodation",    0),
    # Tier 1 — medium risk
    ("Apple Store AU",      "Electronics",      1),
    ("Samsung AU",          "Electronics",      1),
    ("Luxury Watches AU",   "Jewellery",        1),
    ("Goldmark Jewellers",  "Jewellery",        1),
    ("Michael Hill",        "Jewellery",        1),
    ("eBay AU",             "Marketplace",      1),
    ("Gumtree AU",          "Marketplace",      1),
    ("Scoopon",             "Deals",            1),
    ("Catch.com.au",        "Marketplace",      1),
    ("The Iconic",          "Fashion",          1),
    ("ASOS AU",             "Fashion",          1),
    ("Shein AU",            "Fashion",          1),
    ("Wish AU",             "Marketplace",      1),
    ("AliExpress AU",       "Marketplace",      1),
    ("Western Union AU",    "Money Transfer",   1),
    ("MoneyGram AU",        "Money Transfer",   1),
    ("OzForex",             "FX Transfer",      1),
    ("TorFX",               "FX Transfer",      1),
    ("Tabcorp",             "Gambling",         1),
    ("Sportsbet",           "Gambling",         1),
    # Tier 2 — high risk
    ("Crown Casino",        "Gambling",         2),
    ("Star Casino",         "Gambling",         2),
    ("CoinSpot",            "Crypto Exchange",  2),
    ("Independent Reserve", "Crypto Exchange",  2),
    ("Coinbase AU",         "Crypto Exchange",  2),
    ("Swyftx",              "Crypto Exchange",  2),
    ("BTC Markets",         "Crypto Exchange",  2),
    ("Binance AU",          "Crypto Exchange",  2),
    ("FX Global Markets",   "FX Trading",       2),
    ("Plus500 AU",          "CFD Trading",      2),
    ("IG Markets",          "CFD Trading",      2),
    ("CMC Markets",         "CFD Trading",      2),
    ("Unknown Merchant",    "Uncategorised",    2),
    ("Int'l Wire Service",  "Wire Transfer",    2),
    ("Offshore FX AU",      "FX Transfer",      2),
]

ACCOUNT_TYPES = ["personal", "personal", "personal", "business", "joint"]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def build_merchants_df() -> pd.DataFrame:
    rows = []
    for i, (name, category, tier) in enumerate(MERCHANT_CATALOGUE):
        rows.append({
            "merchant_id"      : f"MER-{i+1:04d}",
            "merchant_name"    : name,
            "merchant_category": category,
            "risk_tier"        : tier,
        })
    return pd.DataFrame(rows)


def build_accounts_df(account_ids: list, rng: np.random.Generator) -> pd.DataFrame:
    """Generate account metadata for each unique account."""
    base_date = datetime(2020, 1, 1)
    rows = []
    for aid in account_ids:
        open_date = base_date + timedelta(days=int(rng.integers(0, 4 * 365)))
        rows.append({
            "account_id"       : aid,
            "account_open_date": open_date.strftime("%Y-%m-%d"),
            "account_type"     : rng.choice(ACCOUNT_TYPES),
            "credit_limit"     : round(float(rng.choice([2000, 5000, 10000, 20000, 50000])), 2),
        })
    return pd.DataFrame(rows)


def enrich_transactions(
    df: pd.DataFrame,
    merchants_df: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Add account_id, merchant_id, txn_timestamp to the base dataframe."""
    n = len(df)

    # Assign accounts — ~10 transactions per account on average
    n_accounts = max(1, n // 10)
    account_ids = [f"ACC-{i+1:06d}" for i in range(n_accounts)]
    df["account_id"] = rng.choice(account_ids, size=n)

    # Assign merchants weighted by risk_tier matching the transaction's merchant_risk_tier
    def merchant_ids_for_tier(tier):
        return merchants_df[merchants_df["risk_tier"] == tier]["merchant_id"].tolist()

    tier_to_mids = {0: merchant_ids_for_tier(0),
                    1: merchant_ids_for_tier(1),
                    2: merchant_ids_for_tier(2)}

    merchant_col = []
    for tier in df["merchant_risk_tier"]:
        candidates = tier_to_mids.get(tier, tier_to_mids[0])
        merchant_col.append(rng.choice(candidates))
    df["merchant_id"] = merchant_col

    # Generate timestamps: spread over 90 days, use hour from existing feature
    base_date = datetime(2025, 10, 1)
    day_offsets = rng.integers(0, 90, size=n)
    txn_timestamps = []
    for day_off, hour in zip(day_offsets, df["hour"]):
        minute = int(rng.integers(0, 60))
        second = int(rng.integers(0, 60))
        ts = base_date + timedelta(days=int(day_off), hours=int(hour),
                                   minutes=minute, seconds=second)
        txn_timestamps.append(ts)

    df["txn_timestamp"] = [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in txn_timestamps]
    df["txn_date"]      = [ts.strftime("%Y-%m-%d")          for ts in txn_timestamps]

    return df, account_ids


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def build_db(db_path: Path = DB_PATH) -> None:
    print("=" * 55)
    print("Building SQLite analytics database")
    print("=" * 55)

    rng = np.random.default_rng(42)

    # 1. Load / generate transaction data
    df = load_transactions()
    print(f"Loaded {len(df):,} transactions")

    # 2. Build reference tables
    merchants_df = build_merchants_df()
    print(f"Merchants defined: {len(merchants_df)}")

    # 3. Enrich transactions with FK columns and timestamps
    df, account_ids = enrich_transactions(df, merchants_df, rng)
    accounts_df = build_accounts_df(account_ids, rng)
    print(f"Accounts generated: {len(accounts_df):,}")

    # 4. Write to SQLite
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()

    cur.executescript("""
        DROP TABLE IF EXISTS transactions;
        DROP TABLE IF EXISTS accounts;
        DROP TABLE IF EXISTS merchants;

        CREATE TABLE merchants (
            merchant_id       TEXT PRIMARY KEY,
            merchant_name     TEXT NOT NULL,
            merchant_category TEXT NOT NULL,
            risk_tier         INTEGER NOT NULL
        );

        CREATE TABLE accounts (
            account_id        TEXT PRIMARY KEY,
            account_open_date TEXT NOT NULL,
            account_type      TEXT NOT NULL,
            credit_limit      REAL NOT NULL
        );

        CREATE TABLE transactions (
            transaction_id          TEXT PRIMARY KEY,
            account_id              TEXT NOT NULL,
            merchant_id             TEXT NOT NULL,
            txn_timestamp           TEXT NOT NULL,
            txn_date                TEXT NOT NULL,
            amount                  REAL NOT NULL,
            hour                    INTEGER NOT NULL,
            merchant_risk_tier      INTEGER NOT NULL,
            velocity_1h             INTEGER NOT NULL,
            velocity_24h            INTEGER NOT NULL,
            high_risk_country       INTEGER NOT NULL,
            amount_vs_avg_ratio     REAL NOT NULL,
            days_since_account_open INTEGER NOT NULL,
            is_weekend              INTEGER NOT NULL,
            is_fraud                INTEGER NOT NULL,
            FOREIGN KEY (account_id)  REFERENCES accounts(account_id),
            FOREIGN KEY (merchant_id) REFERENCES merchants(merchant_id)
        );

        CREATE INDEX idx_txns_account   ON transactions(account_id);
        CREATE INDEX idx_txns_merchant  ON transactions(merchant_id);
        CREATE INDEX idx_txns_date      ON transactions(txn_date);
        CREATE INDEX idx_txns_fraud     ON transactions(is_fraud);
    """)

    merchants_df.to_sql("merchants",    conn, if_exists="append", index=False)
    accounts_df.to_sql ("accounts",     conn, if_exists="append", index=False)

    txn_cols = [
        "transaction_id", "account_id", "merchant_id",
        "txn_timestamp", "txn_date",
        "amount", "hour", "merchant_risk_tier",
        "velocity_1h", "velocity_24h", "high_risk_country",
        "amount_vs_avg_ratio", "days_since_account_open",
        "is_weekend", "is_fraud",
    ]
    df[txn_cols].to_sql("transactions", conn, if_exists="append", index=False)

    conn.commit()

    # 5. Verify
    for table in ("merchants", "accounts", "transactions"):
        count = cur.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table:15s}: {count:,} rows")

    conn.close()
    print(f"\nDatabase saved to: {db_path}")
    print("Run: jupyter notebook notebooks/03_sql_analytics.ipynb")


if __name__ == "__main__":
    build_db()
