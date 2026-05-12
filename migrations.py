from src.db import connect, execute, get_backend, identity_column, insert_ignore_sql, table_columns


SCHEMA_VERSION = 3


def _add_column(conn, table, column, definition):
    if column not in table_columns(conn, table):
        execute(conn, f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def _create_predictions_table(conn):
    execute(
        conn,
        f"""
        CREATE TABLE IF NOT EXISTS predictions(
            id {identity_column()},
            user_id INTEGER,
            disease TEXT NOT NULL,
            probability REAL NOT NULL,
            risk TEXT NOT NULL,
            confidence REAL,
            feature_values TEXT,
            actual_result INTEGER,
            clinical_status TEXT DEFAULT 'Pending Review',
            clinical_notes TEXT,
            final_diagnosis TEXT,
            validated_by INTEGER,
            validated_at TIMESTAMP,
            model_version TEXT,
            batch_id TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
    )


def run_migrations(database_path=None):
    conn = connect(database_override=database_path)

    execute(
        conn,
        """
        CREATE TABLE IF NOT EXISTS schema_migrations(
            version INTEGER PRIMARY KEY,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
    )

    _create_predictions_table(conn)
    _add_column(conn, "predictions", "confidence", "REAL")
    _add_column(conn, "predictions", "feature_values", "TEXT")
    _add_column(conn, "predictions", "actual_result", "INTEGER")
    _add_column(conn, "predictions", "clinical_status", "TEXT DEFAULT 'Pending Review'")
    _add_column(conn, "predictions", "clinical_notes", "TEXT")
    _add_column(conn, "predictions", "final_diagnosis", "TEXT")
    _add_column(conn, "predictions", "validated_by", "INTEGER")
    _add_column(conn, "predictions", "validated_at", "TIMESTAMP")
    _add_column(conn, "predictions", "model_version", "TEXT")
    _add_column(conn, "predictions", "batch_id", "TEXT")

    execute(
        conn,
        f"""
        CREATE TABLE IF NOT EXISTS password_reset_tokens(
            id {identity_column()},
            user_id INTEGER NOT NULL,
            token TEXT UNIQUE NOT NULL,
            used INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
    )

    execute(
        conn,
        f"""
        CREATE TABLE IF NOT EXISTS chat_messages(
            id {identity_column()},
            user_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            message TEXT NOT NULL,
            source TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
    )

    execute(
        conn,
        f"""
        CREATE TABLE IF NOT EXISTS appointments(
            id {identity_column()},
            user_id INTEGER NOT NULL,
            doctor_name TEXT NOT NULL,
            specialty TEXT NOT NULL,
            appointment_date TEXT NOT NULL,
            time_slot TEXT NOT NULL,
            mode TEXT DEFAULT 'Video',
            status TEXT DEFAULT 'Booked',
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
    )

    execute(conn, insert_ignore_sql("schema_migrations", "version"), (SCHEMA_VERSION,))
    conn.commit()
    conn.close()
