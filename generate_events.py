"""
Customer Churn Prediction System
Week 1, Day 4: generate 90 days of product activity for the "healthy" 80%
of tenants (steady or growing usage, normal support-ticket volume).

Week 1, Day 5: reuse the same 80/20 cohort split (saved to churn_cohort.json)
to add 90 days of activity for the remaining 20% of tenants -- normal for
the first 60 days, then declining usage and rising support tickets for the
final 30 days.

Safe to re-run: any user who already has events is skipped, so running this
again after Day 4 only fills in the churn cohort, it won't duplicate the
healthy cohort's data.
"""

import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import psycopg2
from psycopg2.extras import execute_values

DB_CONFIG = {
    "host": "localhost",
    "port": 5433,
    "dbname": "customer_intelligence",
    "user": "postgres",
    "password": "churn_dev_pw",
}

NUM_DAYS = 90
CHURN_COHORT_FRACTION = 0.20
COHORT_FILE = Path(__file__).parent / "churn_cohort.json"
COHORT_RANDOM_SEED = 42  # fixed so the 80/20 split is identical every time this file doesn't exist yet

# type, weight, session_duration range (None = no duration)
HEALTHY_EVENT_TYPES = [
    ("login", 0.20, (30, 300)),
    ("dashboard_view", 0.45, (60, 1800)),
    ("api_call", 0.30, None),
    ("support_ticket_created", 0.05, None),
]

DECAY_WINDOW_DAYS = 30  # final N days of the 90-day window where churn tenants decline
DURATION_RANGES = {
    "login": (30, 300),
    "dashboard_view": (60, 1800),
    "api_call": None,
    "support_ticket_created": None,
}


def load_or_create_cohort_split(cursor) -> tuple[list[str], list[str]]:
    """Return (healthy_tenant_ids, churn_tenant_ids). Persists the churn cohort
    so Day 5 operates on the exact same 20% of companies."""
    cursor.execute("SELECT tenant_id FROM tenants ORDER BY tenant_id;")
    all_tenant_ids = [str(row[0]) for row in cursor.fetchall()]

    if COHORT_FILE.exists():
        churn_ids = set(json.loads(COHORT_FILE.read_text())["churn_tenant_ids"])
    else:
        rng = random.Random(COHORT_RANDOM_SEED)
        shuffled = all_tenant_ids[:]
        rng.shuffle(shuffled)
        cutoff = round(len(shuffled) * CHURN_COHORT_FRACTION)
        churn_ids = set(shuffled[:cutoff])
        COHORT_FILE.write_text(json.dumps({"churn_tenant_ids": sorted(churn_ids)}, indent=2))

    healthy_ids = [t for t in all_tenant_ids if t not in churn_ids]
    churn_ids = [t for t in all_tenant_ids if t in churn_ids]
    return healthy_ids, churn_ids


def users_for_tenants(cursor, tenant_ids: list[str]) -> list[str]:
    if not tenant_ids:
        return []
    cursor.execute(
        "SELECT user_id FROM users WHERE tenant_id = ANY(%s::uuid[]);",
        (tenant_ids,),
    )
    return [str(row[0]) for row in cursor.fetchall()]


def weighted_event_type() -> tuple[str, tuple[int, int] | None]:
    types, weights, ranges = zip(*[(t, w, r) for t, w, r in HEALTHY_EVENT_TYPES])
    choice = random.choices(range(len(types)), weights=weights, k=1)[0]
    return types[choice], ranges[choice]


def generate_healthy_events_for_user(user_id: str) -> list[tuple]:
    """Steady-to-slightly-increasing daily activity over the last 90 days."""
    rows = []
    base_activity = random.randint(2, 6)  # this user's typical events/day
    growth = random.uniform(0.0, 0.4)  # some healthy accounts grow, some stay flat

    today = datetime.now(timezone.utc).date()
    for day_offset in range(NUM_DAYS):
        day = today - timedelta(days=NUM_DAYS - 1 - day_offset)
        trend_multiplier = 1 + growth * (day_offset / (NUM_DAYS - 1))
        daily_count = max(0, round(random.gauss(base_activity * trend_multiplier, 1.2)))

        for _ in range(daily_count):
            event_type, duration_range = weighted_event_type()
            event_time = combine_random_time(day)
            duration = random.randint(*duration_range) if duration_range else None
            rows.append((user_id, event_time, event_type, duration))

    return rows


def generate_churn_events_for_user(user_id: str) -> list[tuple]:
    """First 60 days look normal. Final 30 days: engagement (login/dashboard/api)
    decays toward a fraction of baseline, while support tickets ramp up toward
    roughly double baseline -- a company quietly checking out."""
    rows = []
    base_activity = random.randint(2, 6)
    base_rates = {t: base_activity * w for t, w, _ in HEALTHY_EVENT_TYPES}

    today = datetime.now(timezone.utc).date()
    decay_start = NUM_DAYS - DECAY_WINDOW_DAYS  # day_offset where decline begins

    for day_offset in range(NUM_DAYS):
        day = today - timedelta(days=NUM_DAYS - 1 - day_offset)

        if day_offset < decay_start:
            engagement_multiplier = 1.0
            support_multiplier = 1.0
        else:
            decay_progress = (day_offset - decay_start) / (DECAY_WINDOW_DAYS - 1)
            engagement_multiplier = 1 - decay_progress * 0.8  # drops to ~20% of baseline
            support_multiplier = 1 + decay_progress * 1.0  # ramps up to ~2x baseline

        for event_type, rate in base_rates.items():
            multiplier = support_multiplier if event_type == "support_ticket_created" else engagement_multiplier
            daily_count = max(0, round(random.gauss(rate * multiplier, 0.8)))

            duration_range = DURATION_RANGES[event_type]
            for _ in range(daily_count):
                event_time = combine_random_time(day)
                duration = random.randint(*duration_range) if duration_range else None
                rows.append((user_id, event_time, event_type, duration))

    return rows


def users_with_existing_events(cursor) -> set[str]:
    cursor.execute("SELECT DISTINCT user_id FROM product_events;")
    return {str(row[0]) for row in cursor.fetchall()}


def combine_random_time(day):
    seconds_into_day = random.randint(0, 86_399)
    return datetime.combine(day, datetime.min.time()) + timedelta(seconds=seconds_into_day)


def insert_events(cursor, rows: list[tuple]) -> None:
    execute_values(
        cursor,
        """
        INSERT INTO product_events (user_id, "timestamp", event_type, session_duration_secs)
        VALUES %s;
        """,
        rows,
        page_size=5000,
    )


def main():
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        with conn:
            with conn.cursor() as cur:
                healthy_tenant_ids, churn_tenant_ids = load_or_create_cohort_split(cur)
                print(f"Cohort split: {len(healthy_tenant_ids)} healthy tenants, "
                      f"{len(churn_tenant_ids)} at-risk tenants (saved to churn_cohort.json)")

                already_done = users_with_existing_events(cur)

                healthy_user_ids = [u for u in users_for_tenants(cur, healthy_tenant_ids) if u not in already_done]
                churn_user_ids = [u for u in users_for_tenants(cur, churn_tenant_ids) if u not in already_done]

                all_rows = []

                if healthy_user_ids:
                    print(f"Generating 90 days of healthy activity for {len(healthy_user_ids)} users...")
                    for user_id in healthy_user_ids:
                        all_rows.extend(generate_healthy_events_for_user(user_id))
                else:
                    print("Healthy cohort already has events -- skipping.")

                if churn_user_ids:
                    print(f"Generating 90 days of declining activity for {len(churn_user_ids)} at-risk users...")
                    for user_id in churn_user_ids:
                        all_rows.extend(generate_churn_events_for_user(user_id))
                else:
                    print("At-risk cohort already has events -- skipping.")

                if all_rows:
                    print(f"Inserting {len(all_rows)} events...")
                    insert_events(cur, all_rows)

        print(f"Done. Inserted {len(all_rows)} new events.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
