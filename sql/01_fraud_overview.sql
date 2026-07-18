-- ============================================================
-- 01_fraud_overview.sql
-- High-level fraud statistics across the full dataset.
-- Establishes baseline metrics used throughout the analysis.
-- ============================================================

SELECT
    COUNT(*)                                                        AS total_transactions,
    SUM(is_fraud)                                                   AS fraud_count,
    COUNT(*) - SUM(is_fraud)                                        AS legitimate_count,
    ROUND(AVG(is_fraud) * 100, 3)                                  AS fraud_rate_pct,

    -- Volume metrics
    ROUND(SUM(amount), 2)                                           AS total_volume_aud,
    ROUND(SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END), 2)   AS fraud_volume_aud,
    ROUND(
        SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END) * 100.0
        / SUM(amount),
        2
    )                                                               AS fraud_volume_pct,

    -- Amount profiles
    ROUND(AVG(CASE WHEN is_fraud = 0 THEN amount END), 2)          AS avg_legit_amount,
    ROUND(AVG(CASE WHEN is_fraud = 1 THEN amount END), 2)          AS avg_fraud_amount,
    ROUND(AVG(CASE WHEN is_fraud = 1 THEN amount END)
        / AVG(CASE WHEN is_fraud = 0 THEN amount END), 2)          AS fraud_to_legit_amount_ratio,

    -- Velocity profiles
    ROUND(AVG(CASE WHEN is_fraud = 0 THEN velocity_1h END), 2)     AS avg_legit_velocity_1h,
    ROUND(AVG(CASE WHEN is_fraud = 1 THEN velocity_1h END), 2)     AS avg_fraud_velocity_1h,

    -- Risk flag rates
    ROUND(AVG(CASE WHEN is_fraud = 1 THEN high_risk_country END) * 100, 1)
                                                                    AS fraud_high_risk_country_pct,
    ROUND(AVG(CASE WHEN is_fraud = 0 THEN high_risk_country END) * 100, 1)
                                                                    AS legit_high_risk_country_pct
FROM transactions;
