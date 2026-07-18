-- ============================================================
-- 03_velocity_analysis.sql
-- Account-level velocity: rapid successive transactions.
-- Demonstrates: LAG/LEAD, PARTITION BY, date arithmetic,
-- self-join alternative using window functions.
-- Detects ATO (account takeover) and CNP burst patterns.
-- ============================================================

WITH account_txn_sequence AS (
    SELECT
        transaction_id,
        account_id,
        txn_timestamp,
        amount,
        merchant_id,
        high_risk_country,
        is_fraud,

        -- Previous transaction for this account
        LAG(txn_timestamp) OVER (
            PARTITION BY account_id ORDER BY txn_timestamp
        )                                                           AS prev_txn_timestamp,

        LAG(amount) OVER (
            PARTITION BY account_id ORDER BY txn_timestamp
        )                                                           AS prev_amount,

        LAG(merchant_id) OVER (
            PARTITION BY account_id ORDER BY txn_timestamp
        )                                                           AS prev_merchant_id,

        -- Next transaction for this account (look-ahead)
        LEAD(txn_timestamp) OVER (
            PARTITION BY account_id ORDER BY txn_timestamp
        )                                                           AS next_txn_timestamp,

        -- Running count within account (transaction sequence number)
        ROW_NUMBER() OVER (
            PARTITION BY account_id ORDER BY txn_timestamp
        )                                                           AS txn_seq_num,

        -- Rank by amount within account (1 = largest transaction)
        RANK() OVER (
            PARTITION BY account_id ORDER BY amount DESC
        )                                                           AS amount_rank_in_account

    FROM transactions
),

enriched AS (
    SELECT
        s.*,
        -- Minutes elapsed since the previous transaction on this account
        ROUND(
            (JULIANDAY(s.txn_timestamp) - JULIANDAY(s.prev_txn_timestamp)) * 24 * 60,
            1
        )                                                           AS minutes_since_prev_txn,

        -- Amount change vs previous transaction
        ROUND(s.amount / NULLIF(s.prev_amount, 0), 2)             AS amount_ratio_vs_prev,

        -- Flag: same account transacted at a different merchant within 10 minutes
        CASE
            WHEN (JULIANDAY(s.txn_timestamp) - JULIANDAY(s.prev_txn_timestamp)) * 24 * 60 < 10
             AND s.merchant_id != s.prev_merchant_id
            THEN 1 ELSE 0
        END                                                         AS multi_merchant_burst_flag

    FROM account_txn_sequence s
    WHERE s.prev_txn_timestamp IS NOT NULL
)

SELECT
    e.transaction_id,
    e.account_id,
    e.txn_timestamp,
    ROUND(e.amount, 2)                  AS amount,
    ROUND(e.prev_amount, 2)             AS prev_amount,
    e.minutes_since_prev_txn,
    e.amount_ratio_vs_prev,
    e.multi_merchant_burst_flag,
    e.txn_seq_num,
    e.amount_rank_in_account,
    e.high_risk_country,
    e.is_fraud
FROM enriched e
WHERE e.minutes_since_prev_txn < 60   -- transactions within 1 hour of a prior transaction
ORDER BY e.minutes_since_prev_txn ASC, e.amount DESC
LIMIT 100;
