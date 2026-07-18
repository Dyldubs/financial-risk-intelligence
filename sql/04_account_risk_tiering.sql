-- ============================================================
-- 04_account_risk_tiering.sql
-- Rule-based account risk tiering using multi-step CTEs.
-- Demonstrates: chained CTEs, conditional aggregation,
-- CASE WHEN scoring, derived risk tier classification.
-- Mirrors how bank risk teams build account-level watchlists.
-- ============================================================

WITH account_metrics AS (
    -- Step 1: aggregate raw metrics per account
    SELECT
        t.account_id,
        a.account_type,
        a.credit_limit,
        COUNT(t.transaction_id)                                     AS total_txns,
        SUM(t.is_fraud)                                             AS confirmed_fraud_count,
        ROUND(SUM(t.amount), 2)                                     AS total_spend_aud,
        ROUND(AVG(t.amount), 2)                                     AS avg_txn_amount,
        ROUND(MAX(t.amount), 2)                                     AS max_txn_amount,
        MAX(t.velocity_1h)                                          AS peak_velocity_1h,
        MAX(t.velocity_24h)                                         AS peak_velocity_24h,
        SUM(t.high_risk_country)                                    AS high_risk_country_txns,
        ROUND(AVG(t.amount_vs_avg_ratio), 2)                        AS avg_amount_ratio,
        MAX(t.amount_vs_avg_ratio)                                  AS max_amount_ratio,
        SUM(CASE WHEN t.merchant_risk_tier = 2 THEN 1 ELSE 0 END)  AS high_risk_merchant_txns,
        COUNT(DISTINCT t.merchant_id)                               AS distinct_merchants,
        MIN(t.txn_date)                                             AS first_txn_date,
        MAX(t.txn_date)                                             AS last_txn_date
    FROM transactions t
    JOIN accounts a ON t.account_id = a.account_id
    GROUP BY t.account_id, a.account_type, a.credit_limit
),

account_risk_scores AS (
    -- Step 2: compute a weighted point score for each risk signal
    SELECT
        *,
        (
            -- Confirmed fraud: highest weight
            CASE WHEN confirmed_fraud_count >= 2 THEN 50
                 WHEN confirmed_fraud_count =  1 THEN 30
                 ELSE 0
            END
            -- High-risk jurisdiction exposure
          + CASE WHEN high_risk_country_txns >= 3  THEN 25
                 WHEN high_risk_country_txns >= 1  THEN 15
                 ELSE 0
            END
            -- Burst velocity
          + CASE WHEN peak_velocity_1h >= 6  THEN 20
                 WHEN peak_velocity_1h >= 4  THEN 12
                 WHEN peak_velocity_1h >= 2  THEN 5
                 ELSE 0
            END
            -- Unusual spend vs historical average
          + CASE WHEN max_amount_ratio >= 10  THEN 15
                 WHEN max_amount_ratio >=  5  THEN 10
                 WHEN max_amount_ratio >=  3  THEN  5
                 ELSE 0
            END
            -- High-risk merchant usage
          + CASE WHEN high_risk_merchant_txns >= 3  THEN 10
                 WHEN high_risk_merchant_txns >= 1  THEN  5
                 ELSE 0
            END
        )                                                           AS risk_score
    FROM account_metrics
),

tiered AS (
    -- Step 3: map score to tier
    SELECT
        *,
        CASE
            WHEN risk_score >= 60 THEN 'CRITICAL'
            WHEN risk_score >= 35 THEN 'HIGH'
            WHEN risk_score >= 15 THEN 'MEDIUM'
            ELSE 'LOW'
        END                                                         AS account_risk_tier,

        -- Percentile rank (higher = riskier relative to all accounts)
        ROUND(
            PERCENT_RANK() OVER (ORDER BY risk_score) * 100,
            1
        )                                                           AS risk_percentile
    FROM account_risk_scores
)

SELECT
    account_id,
    account_type,
    account_risk_tier,
    risk_score,
    risk_percentile,
    total_txns,
    confirmed_fraud_count,
    total_spend_aud,
    peak_velocity_1h,
    high_risk_country_txns,
    high_risk_merchant_txns,
    ROUND(max_amount_ratio, 1)          AS max_amount_ratio,
    credit_limit,
    first_txn_date,
    last_txn_date
FROM tiered
ORDER BY risk_score DESC, confirmed_fraud_count DESC;
