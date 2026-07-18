-- ============================================================
-- 07_alert_queue.sql
-- Prioritised analyst alert queue for high-risk transactions.
-- Demonstrates: multi-table JOIN, composite scoring with
-- CASE WHEN, subquery for account context, NTILE bucketing.
-- This is the query that would feed a real-time alert dashboard.
-- ============================================================

WITH account_baselines AS (
    -- Pre-compute per-account averages to use as a baseline
    SELECT
        account_id,
        ROUND(AVG(amount), 2)   AS account_avg_amount,
        COUNT(*)                AS account_total_txns,
        SUM(is_fraud)           AS account_prior_fraud_count
    FROM transactions
    GROUP BY account_id
),

alert_candidates AS (
    SELECT
        t.transaction_id,
        t.account_id,
        t.txn_timestamp,
        t.txn_date,
        ROUND(t.amount, 2)                      AS amount,
        m.merchant_name,
        m.merchant_category,
        m.risk_tier                             AS merchant_risk_tier,
        t.velocity_1h,
        t.velocity_24h,
        t.high_risk_country,
        ROUND(t.amount_vs_avg_ratio, 2)         AS amount_vs_avg_ratio,
        t.days_since_account_open,
        t.is_weekend,
        t.is_fraud,
        b.account_avg_amount,
        b.account_prior_fraud_count,

        -- Composite alert priority score (higher = review first)
        (
            CASE WHEN t.amount >= 10000               THEN 30
                 WHEN t.amount >=  5000               THEN 20
                 WHEN t.amount >=  2000               THEN 10
                 ELSE 0
            END
          + CASE WHEN t.high_risk_country = 1         THEN 25 ELSE 0 END
          + CASE WHEN t.velocity_1h >= 5              THEN 20
                 WHEN t.velocity_1h >= 3              THEN 10
                 ELSE 0
            END
          + CASE WHEN t.amount_vs_avg_ratio >= 10     THEN 20
                 WHEN t.amount_vs_avg_ratio >=  5     THEN 12
                 WHEN t.amount_vs_avg_ratio >=  3     THEN  6
                 ELSE 0
            END
          + CASE WHEN m.risk_tier = 2                 THEN 15
                 WHEN m.risk_tier = 1                 THEN  5
                 ELSE 0
            END
          + CASE WHEN t.days_since_account_open <= 30 THEN 10 ELSE 0 END
          + CASE WHEN b.account_prior_fraud_count > 0 THEN 15 ELSE 0 END
          + CASE WHEN t.velocity_24h >= 15            THEN 10 ELSE 0 END
        )                                               AS alert_priority_score

    FROM transactions t
    JOIN merchants      m ON t.merchant_id  = m.merchant_id
    JOIN account_baselines b ON t.account_id = b.account_id
    WHERE
        -- Only surface transactions with at least one risk flag
        t.amount >= 1000
        AND (
               t.high_risk_country  = 1
            OR t.velocity_1h       >= 3
            OR t.amount_vs_avg_ratio >= 3
            OR m.risk_tier          = 2
            OR t.days_since_account_open <= 30
        )
),

final AS (
    SELECT
        *,
        -- Bucket into 4 review priority bands
        NTILE(4) OVER (ORDER BY alert_priority_score DESC)         AS priority_band,
        -- Rank within the queue
        ROW_NUMBER() OVER (ORDER BY alert_priority_score DESC)     AS queue_position
    FROM alert_candidates
)

SELECT
    queue_position,
    CASE priority_band
        WHEN 1 THEN 'P1 — Immediate'
        WHEN 2 THEN 'P2 — Same Day'
        WHEN 3 THEN 'P3 — Next Day'
        WHEN 4 THEN 'P4 — Weekly'
    END                                                             AS review_sla,
    alert_priority_score,
    transaction_id,
    account_id,
    txn_timestamp,
    amount,
    merchant_name,
    merchant_category,
    CASE merchant_risk_tier
        WHEN 0 THEN 'Low' WHEN 1 THEN 'Medium' WHEN 2 THEN 'High'
    END                                                             AS merchant_risk,
    velocity_1h,
    high_risk_country,
    amount_vs_avg_ratio,
    days_since_account_open,
    account_prior_fraud_count,
    is_fraud                                                        AS confirmed_fraud
FROM final
ORDER BY queue_position
LIMIT 50;
