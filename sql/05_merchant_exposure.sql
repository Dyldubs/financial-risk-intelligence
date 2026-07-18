-- ============================================================
-- 05_merchant_exposure.sql
-- Merchant-level fraud exposure and volume ranking.
-- Demonstrates: JOIN, conditional aggregation, HAVING,
-- ratio calculations, multi-column ORDER BY.
-- Identifies high-risk merchants for enhanced due diligence.
-- ============================================================

WITH merchant_stats AS (
    SELECT
        m.merchant_id,
        m.merchant_name,
        m.merchant_category,
        m.risk_tier,

        COUNT(t.transaction_id)                                         AS total_txns,
        SUM(t.is_fraud)                                                 AS fraud_txns,
        COUNT(t.transaction_id) - SUM(t.is_fraud)                      AS legit_txns,

        ROUND(AVG(t.is_fraud) * 100, 2)                                AS fraud_rate_pct,

        ROUND(SUM(t.amount), 2)                                         AS total_volume_aud,
        ROUND(SUM(CASE WHEN t.is_fraud = 1 THEN t.amount ELSE 0 END), 2)
                                                                        AS fraud_volume_aud,
        ROUND(SUM(CASE WHEN t.is_fraud = 0 THEN t.amount ELSE 0 END), 2)
                                                                        AS legit_volume_aud,

        ROUND(
            SUM(CASE WHEN t.is_fraud = 1 THEN t.amount ELSE 0 END) * 100.0
            / NULLIF(SUM(t.amount), 0),
            2
        )                                                               AS fraud_volume_pct,

        ROUND(AVG(t.amount), 2)                                         AS avg_txn_amount,
        ROUND(MAX(t.amount), 2)                                         AS max_txn_amount,
        COUNT(DISTINCT t.account_id)                                    AS distinct_accounts

    FROM merchants m
    LEFT JOIN transactions t ON m.merchant_id = t.merchant_id
    GROUP BY m.merchant_id, m.merchant_name, m.merchant_category, m.risk_tier
    HAVING total_txns >= 5
),

ranked AS (
    SELECT
        *,
        -- Rank by fraud volume (most exposed merchants first)
        RANK() OVER (ORDER BY fraud_volume_aud DESC)                    AS fraud_volume_rank,
        -- Rank by fraud rate within each risk tier
        RANK() OVER (
            PARTITION BY risk_tier ORDER BY fraud_rate_pct DESC
        )                                                               AS fraud_rate_rank_in_tier
    FROM merchant_stats
)

SELECT
    fraud_volume_rank,
    merchant_name,
    merchant_category,
    CASE risk_tier
        WHEN 0 THEN 'Low'
        WHEN 1 THEN 'Medium'
        WHEN 2 THEN 'High'
    END                                                                 AS merchant_risk_tier,
    total_txns,
    fraud_txns,
    fraud_rate_pct,
    fraud_volume_aud,
    fraud_volume_pct,
    total_volume_aud,
    avg_txn_amount,
    distinct_accounts,
    fraud_rate_rank_in_tier
FROM ranked
ORDER BY fraud_volume_aud DESC
LIMIT 25;
