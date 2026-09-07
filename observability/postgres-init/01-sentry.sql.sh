#!/bin/bash
set -e
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE USER sentry WITH PASSWORD 'sentry';
    CREATE DATABASE sentry OWNER sentry;
    GRANT ALL PRIVILEGES ON DATABASE sentry TO sentry;
EOSQL