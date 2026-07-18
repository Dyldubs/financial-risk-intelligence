-- ============================================================
-- 06_time_of_day_analysis.sql
-- Fraud patterns by hour of day and time window.
-- Demonstrates: CASE WHEN categorisation, aggregation with
-- multiple group-by levels, ratio to overall average.
-- Informs the is_night feature used in the ML model.
-- ============================================================

WITH hourly AS (
    SELECT
        hour,
        CASE
            WHEN hour BETWEEN  6 AND 11 THEN '1_Morning (6am–12pm)'
            WHEN hour BETWEEN 12 AND 17 THEN '2_Afternoon (12pm–6pm)'
            WHEN hour BETWEEN 18 AND 22 THEN '3_Evening (6pm–11pm)'
            ELSE                              '4_Night (11pm–6am)'
        END                                                         AS time_window,
        COUNT(*)                                                    AS total_txns,
        SUM(is_fraud)                                               AS fraud_txns,
        ROUND(AVG(is_fraud) * 100, 3)                              AS fraud_rate_pct,
        ROUND(AVG(amount), 2)                                       AS avg_amount,
        ROUND(SUM(amount), 2)                                       AS total_volume_aud,
        ROUND(
            SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END), 2
        )                                                           AS fraud_volume_aud
    FROM transactions
    GROUP BY hour
),

overall AS (
    SELECT AVG(is_fraud) * 100 AS overall_fraud_rate_pct
    FROM transactions
)

SELECT
    h.hour,
    h.time_window,
    h.total_txns,
    h.fraud_txns,
    h.fraud_rate_pct,
    -- Uplift vs overall average (>1 = riskier than average)
    ROUND(h.fraud_rate_pct / o.overall_fraud_rate_pct, 2)          AS fraud_rate_uplift,
    h.avg_amount,
    h.total_volume_aud,
    h.fraud_volume_aud,
    -- Share of total daily transactions occurring in this hour
    ROUND(h.total_txns * 100.0 / SUM(h.total_txns) OVER (), 1)    AS txn_share_pct
FROM hourly h
CROSS JOIN overall o
ORDER BY h.hour;
