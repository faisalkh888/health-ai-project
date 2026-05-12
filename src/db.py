import os
import sqlite3
from urllib.parse import parse_qs, unquote, urlparse

from config import DATABASE, DATABASE_URL


def get_backend():
    if DATABASE_URL.startswith("postgres://") or DATABASE_URL.startswith("postgresql://"):
        return "postgresql"
    if DATABASE_URL.startswith("mysql://"):
        return "mysql"
    return "sqlite"


def _normalize_sql(sql):
    if get_backend() == "sqlite":
        return sql
    return sql.replace("?", "%s")


def connect(database_override=None, database_url=None):
    if database_override:
        conn = sqlite3.connect(database_override)
        conn.row_factory = sqlite3.Row
        return conn

    active_url = database_url or DATABASE_URL
    backend = get_backend() if active_url == DATABASE_URL else (
        "postgresql" if active_url.startswith(("postgres://", "postgresql://")) else
        "mysql" if active_url.startswith("mysql://") else
        "sqlite"
    )
    if backend == "sqlite":
        conn = sqlite3.connect(DATABASE)
        conn.row_factory = sqlite3.Row
        return conn

    parsed = urlparse(active_url)
    database = unquote((parsed.path or "").lstrip("/"))
    params = parse_qs(parsed.query or "")

    if backend == "postgresql":
        import psycopg2
        from psycopg2.extras import RealDictCursor

        return psycopg2.connect(
            dbname=database,
            user=unquote(parsed.username or ""),
            password=unquote(parsed.password or ""),
            host=parsed.hostname or "localhost",
            port=parsed.port or 5432,
            sslmode=params.get("sslmode", ["prefer"])[0],
            cursor_factory=RealDictCursor,
        )

    import pymysql

    return pymysql.connect(
        host=parsed.hostname or "localhost",
        user=unquote(parsed.username or ""),
        password=unquote(parsed.password or ""),
        database=database,
        port=parsed.port or 3306,
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=False,
    )


def _row_to_dict(row):
    if row is None:
        return None
    if isinstance(row, dict):
        return row
    if isinstance(row, sqlite3.Row):
        return dict(row)
    if hasattr(row, "_asdict"):
        return row._asdict()
    return row


def execute(conn, sql, params=()):
    cursor = conn.cursor()
    cursor.execute(_normalize_sql(sql), params)
    return cursor


def fetchone(cursor):
    return _row_to_dict(cursor.fetchone())


def fetchall(cursor):
    return [_row_to_dict(row) for row in cursor.fetchall()]


def fetch_value(conn, sql, params=()):
    row = fetchone(execute(conn, sql, params))
    if row is None:
        return None
    if isinstance(row, dict):
        return next(iter(row.values()))
    return row[0]


def last_insert_id(cursor, table_name=None):
    backend = get_backend()
    if backend == "sqlite":
        return cursor.lastrowid
    if backend == "postgresql":
        row = cursor.fetchone()
        if isinstance(row, dict):
            return next(iter(row.values()))
        return row[0]
    return cursor.lastrowid


def identity_column():
    backend = get_backend()
    if backend == "postgresql":
        return "INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY"
    if backend == "mysql":
        return "INTEGER PRIMARY KEY AUTO_INCREMENT"
    return "INTEGER PRIMARY KEY AUTOINCREMENT"


def current_timestamp_sql():
    return "CURRENT_TIMESTAMP"


def table_columns(conn, table):
    backend = get_backend()
    if backend == "sqlite":
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table})")
        return {row["name"] for row in cursor.fetchall()}

    if backend == "postgresql":
        row_sql = """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = %s
        """
    else:
        row_sql = """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = DATABASE() AND table_name = %s
        """
    cursor = conn.cursor()
    cursor.execute(row_sql, (table,))
    return {row["column_name"] if isinstance(row, dict) else row[0] for row in cursor.fetchall()}


def insert_ignore_sql(table, column):
    backend = get_backend()
    if backend == "postgresql":
        return f"INSERT INTO {table}({column}) VALUES (%s) ON CONFLICT DO NOTHING"
    if backend == "mysql":
        return f"INSERT IGNORE INTO {table}({column}) VALUES (%s)"
    return f"INSERT OR IGNORE INTO {table}({column}) VALUES (?)"
