"""
Customer Churn Prediction System
Week 1, Day 3: generate fake companies (tenants) and fake users.
"""

import random
from datetime import date, timedelta

import psycopg2
from faker import Faker

DB_CONFIG = {
    "host": "localhost",
    "port": 5433,
    "dbname": "customer_intelligence",
    "user": "postgres",
    "password": "churn_dev_pw",
}

NUM_TENANTS = 100
NUM_USERS = 500

SUBSCRIPTION_TIERS = ["Starter", "Pro", "Enterprise"]

fake = Faker()


def random_signup_date() -> date:
    days_ago = random.randint(30, 900)
    return date.today() - timedelta(days=days_ago)


def generate_tenants(cursor, n: int) -> list[str]:
    tenant_ids = []
    for _ in range(n):
        tier = random.choice(SUBSCRIPTION_TIERS)
        contract_value = {
            "Starter": random.uniform(1_000, 10_000),
            "Pro": random.uniform(10_000, 50_000),
            "Enterprise": random.uniform(50_000, 250_000),
        }[tier]

        cursor.execute(
            """
            INSERT INTO tenants (company_name, subscription_tier, contract_value, signup_date)
            VALUES (%s, %s, %s, %s)
            RETURNING tenant_id;
            """,
            (fake.company(), tier, round(contract_value, 2), random_signup_date()),
        )
        tenant_ids.append(cursor.fetchone()[0])
    return tenant_ids


def generate_users(cursor, tenant_ids: list[str], n: int) -> list[tuple[str, str]]:
    """Randomly distribute n users across the given tenants. Every tenant gets at least one."""
    user_ids = []

    # guarantee each tenant has at least one user
    for tenant_id in tenant_ids:
        cursor.execute(
            """
            INSERT INTO users (tenant_id, email, created_at)
            VALUES (%s, %s, %s)
            RETURNING user_id;
            """,
            (tenant_id, fake.unique.email(), fake.date_time_between(start_date="-2y")),
        )
        user_ids.append((cursor.fetchone()[0], tenant_id))

    remaining = n - len(tenant_ids)
    for _ in range(remaining):
        tenant_id = random.choice(tenant_ids)
        cursor.execute(
            """
            INSERT INTO users (tenant_id, email, created_at)
            VALUES (%s, %s, %s)
            RETURNING user_id;
            """,
            (tenant_id, fake.unique.email(), fake.date_time_between(start_date="-2y")),
        )
        user_ids.append((cursor.fetchone()[0], tenant_id))

    return user_ids


def main():
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        with conn:
            with conn.cursor() as cur:
                print(f"Generating {NUM_TENANTS} tenants...")
                tenant_ids = generate_tenants(cur, NUM_TENANTS)

                print(f"Generating {NUM_USERS} users across {len(tenant_ids)} tenants...")
                user_ids = generate_users(cur, tenant_ids, NUM_USERS)

        print(f"Done. Inserted {len(tenant_ids)} tenants and {len(user_ids)} users.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
