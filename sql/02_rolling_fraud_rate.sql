-- ============================================================
-- 02_rolling_fraud_rate.sql
-- Daily fraud rate with 7-day rolling average.
-- Demonstrates: window functions (SUM OVER, ROWS BETWEEN),
-- date aggregation, trend analysis.
-- ============================================================

WITH daily_stats AS (
    SELECT
        txn_date,
        COUNT(*)        AS daily_txns,
        SUM(is_fraud)   AS daily_fraud,
        ROUND(SUM(amount), 2) AS daily_volume_aud
    FROM transactions
    GROUP BY txn_date
)

SELECT
    txn_date,
    daily_txns,
    daily_fraud,
    ROUND(daily_fraud * 100.0 / daily_txns, 2)                     AS daily_fraud_rate_pct,
    daily_volume_aud,

    -- 7-day rolling totals
    SUM(daily_txns)  OVER (
        ORDER BY txn_date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    )                                                               AS rolling_7d_txns,

    SUM(daily_fraud) OVER (
        ORDER BY txn_date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    )                                                               AS rolling_7d_fraud,

    ROUND(
        SUM(daily_fraud) OVER (
            ORDER BY txn_date
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) * 100.0
        / SUM(daily_txns) OVER (
            ORDER BY txn_date
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ),
        3
    )                                                               AS rolling_7d_fraud_rate_pct,

    -- 7-day rolling fraud volume
    ROUND(
        SUM(daily_volume_aud) OVER (
            ORDER BY txn_date
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ),
        2
    )                                                               AS rolling_7d_volume_aud,

    -- Cumulative fraud count (running total)
    SUM(daily_fraud) OVER (
        ORDER BY txn_date
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    )                                                               AS cumulative_fraud_count

FROM daily_stats
ORDER BY txn_date;
