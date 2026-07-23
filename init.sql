-- Customer Churn Prediction System
-- Week 1, Day 2: Relational schema

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE tenants (
    tenant_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    company_name VARCHAR(255) NOT NULL,
    subscription_tier VARCHAR(50) NOT NULL,
    contract_value NUMERIC(12, 2) NOT NULL,
    signup_date DATE NOT NULL
);

CREATE TABLE users (
    user_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES tenants(tenant_id),
    email VARCHAR(255) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT now()
);

CREATE TABLE product_events (
    event_id BIGSERIAL PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(user_id),
    "timestamp" TIMESTAMP NOT NULL,
    event_type VARCHAR(50) NOT NULL,
    session_duration_secs INT
);

CREATE INDEX idx_users_tenant_id ON users(tenant_id);
CREATE INDEX idx_product_events_timestamp ON product_events("timestamp");
