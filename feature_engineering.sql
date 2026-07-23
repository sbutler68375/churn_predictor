-- Customer Churn Prediction System
-- Week 2, Day 4: wrap the full pipeline in a permanent view, collapsed to one row per
-- tenant (each company's most recent snapshot), with a binary is_churned label attached.
-- Threshold of 0.15 on engagement_velocity was chosen by eyeballing known at-risk vs.
-- healthy companies during Day 3 -- see the validation query below the CREATE VIEW block.

CREATE OR REPLACE VIEW v_ml_churn_features AS

WITH tenant_events AS (
    SELECT
        t.tenant_id,
        t.company_name,
        pe.event_id,
        pe."timestamp",
        pe.event_type,
        pe.session_duration_secs
    FROM product_events pe
    JOIN users u ON u.user_id = pe.user_id
    JOIN tenants t ON t.tenant_id = u.tenant_id
),

-- Week 2, Day 2: rolling activity counts per event, "as of" that event's own timestamp.
-- These are the raw ingredients Day 3's velocity calculation will compare against each other.
tenant_rolling_activity AS (
    SELECT
        te.*,
        COUNT(event_id) OVER (
            PARTITION BY tenant_id
            ORDER BY "timestamp"
            RANGE BETWEEN INTERVAL '7 days' PRECEDING AND CURRENT ROW
        ) AS events_last_7d,
        COUNT(event_id) OVER (
            PARTITION BY tenant_id
            ORDER BY "timestamp"
            RANGE BETWEEN INTERVAL '30 days' PRECEDING AND CURRENT ROW
        ) AS events_last_30d,
        COUNT(event_id) OVER (
            PARTITION BY tenant_id
            ORDER BY "timestamp"
            RANGE BETWEEN INTERVAL '90 days' PRECEDING AND CURRENT ROW
        ) AS events_last_90d,
        COUNT(event_id) FILTER (WHERE event_type = 'support_ticket_created') OVER (
            PARTITION BY tenant_id
            ORDER BY "timestamp"
            RANGE BETWEEN INTERVAL '30 days' PRECEDING AND CURRENT ROW
        ) AS support_tickets_last_30d
    FROM tenant_events te
),

-- Week 2, Day 3: engagement velocity -- how far recent (30d) daily activity has drifted
-- from this tenant's own 90-day baseline daily activity. Normalized to a per-day rate first
-- since the two windows are different sizes and their raw totals aren't directly comparable.
-- Positive = declining (bad sign), ~0 = no change, negative = growing (good sign).
tenant_velocity AS (
    SELECT
        tra.*,
        (events_last_30d::numeric / 30) AS events_per_day_30d,
        (events_last_90d::numeric / 90) AS events_per_day_90d,
        CASE
            WHEN events_last_90d = 0 THEN NULL
            ELSE ROUND(
                (
                    (events_last_90d::numeric / 90) - (events_last_30d::numeric / 30)
                ) / (events_last_90d::numeric / 90),
                4
            )
        END AS engagement_velocity
    FROM tenant_rolling_activity tra
),

-- Week 2, Day 4: keep only each tenant's most recent event -- their latest, most complete
-- snapshot of rolling activity and velocity -- to collapse from "one row per event" to
-- "one row per tenant."
tenant_latest_snapshot AS (
    SELECT
        tv.*,
        ROW_NUMBER() OVER (PARTITION BY tenant_id ORDER BY "timestamp" DESC) AS rn
    FROM tenant_velocity tv
)

SELECT
    tenant_id,
    company_name,
    events_last_7d,
    events_last_30d,
    events_last_90d,
    support_tickets_last_30d,
    engagement_velocity,
    CASE WHEN engagement_velocity > 0.15 THEN 1 ELSE 0 END AS is_churned
FROM tenant_latest_snapshot
WHERE rn = 1;
