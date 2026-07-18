-- ============================================================
-- 08_account_cohort_analysis.sql
-- Fraud rate and spend behaviour by account age cohort.
-- Demonstrates: CTE + JOIN, CASE WHEN bucketing, cohort
-- aggregation, multiple metrics in one pass.
-- New accounts (<30 days) consistently show higher fraud rates
-- due to first-party fraud and synthetic identity patterns.
-- ============================================================

WITH cohort_map AS (
    -- Assign each account a single cohort bucket based on account age
    -- at the time of its transactions (using the median days value)
    SELECT
        account_id,
        ROUND(AVG(days_since_account_open))                         AS avg_days_open,
        CASE
            WHEN AVG(days_since_account_open) <=  30 THEN 1
            WHEN AVG(days_since_account_open) <=  90 THEN 2
            WHEN AVG(days_since_account_open) <= 365 THEN 3
            WHEN AVG(days_since_account_open) <= 730 THEN 4
            ELSE                                           5
        END                                                         AS cohort_id,
        CASE
            WHEN AVG(days_since_account_open) <=  30 THEN '1 — New (≤30 days)'
            WHEN AVG(days_since_account_open) <=  90 THEN '2 — Early (31–90 days)'
            WHEN AVG(days_since_account_open) <= 365 THEN '3 — Established (91d–1yr)'
            WHEN AVG(days_since_account_open) <= 730 THEN '4 — Mature (1–2 years)'
            ELSE                                           '5 — Veteran (2+ years)'
        END                                                         AS cohort_label
    FROM transactions
    GROUP BY account_id
),

cohort_txns AS (
    SELECT
        c.cohort_id,
        c.cohort_label,
        t.transaction_id,
        t.amount,
        t.is_fraud,
        t.high_risk_country,
        t.merchant_risk_tier,
        t.velocity_1h
    FROM transactions t
    JOIN cohort_map c ON t.account_id = c.account_id
)

SELECT
    cohort_label,
    COUNT(DISTINCT (
        SELECT account_id FROM transactions t2
        JOIN cohort_map c2 ON t2.account_id = c2.account_id
        WHERE c2.cohort_id = ct.cohort_id
        LIMIT 1
    ))                                                              AS approx_accounts,

    COUNT(transaction_id)                                           AS total_txns,
    SUM(is_fraud)                                                   AS fraud_txns,
    ROUND(AVG(is_fraud) * 100, 2)                                  AS fraud_rate_pct,

    -- Spend profile
    ROUND(AVG(amount), 2)                                           AS avg_txn_amount,
    ROUND(SUM(amount), 2)                                           AS total_volume_aud,
    ROUND(SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END), 2)   AS fraud_volume_aud,

    -- Risk signal prevalence within cohort
    ROUND(AVG(high_risk_country) * 100, 1)                         AS high_risk_country_pct,
    ROUND(AVG(CASE WHEN merchant_risk_tier = 2 THEN 1.0 ELSE 0 END) * 100, 1)
                                                                    AS high_risk_merchant_pct,
    ROUND(AVG(velocity_1h), 2)                                     AS avg_velocity_1h,

    -- Fraud rate uplift vs overall average
    ROUND(
        AVG(is_fraud) * 100 / (SELECT AVG(is_fraud) * 100 FROM transactions),
        2
    )                                                               AS fraud_rate_uplift

FROM cohort_txns ct
GROUP BY cohort_id, cohort_label
ORDER BY cohort_id;
