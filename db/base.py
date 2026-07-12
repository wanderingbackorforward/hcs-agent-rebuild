"""Session manager."""
import os
import logging
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker, Session
from config.database import db_config

logger = logging.getLogger(__name__)


class SessionManager:
    def __init__(self, db_path: str = None):
        db_path = db_path or db_config.connection_string
        connect_args = {}
        if "sqlite" in db_path:
            connect_args["check_same_thread"] = False
            db_file = db_path.replace("sqlite:///", "")
            db_dir = os.path.dirname(db_file)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir, exist_ok=True)

        self.engine = create_engine(
            db_path, connect_args=connect_args, **db_config.get_engine_kwargs()
        )
        self.SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine
        )

    def get_session(self) -> Session:
        return self.SessionLocal()

    def close(self):
        self.engine.dispose()


def run_lightweight_migrations(engine):
    """Add new columns to existing SQLite tables (create_all won't do this).

    Called once at startup after create_all. Each column is added only if the
    table exists and the column is missing. Safe to call repeatedly.
    """
    inspector = inspect(engine)
    if not inspector.has_table("knowledge_documents"):
        return  # table doesn't exist yet; create_all will handle it

    existing = {c["name"] for c in inspector.get_columns("knowledge_documents")}
    migrations = [
        ("archive_path", "VARCHAR"),
        ("version", "INTEGER DEFAULT 1"),
    ]
    with engine.connect() as conn:
        for col_name, col_type in migrations:
            if col_name not in existing:
                conn.execute(text(
                    f"ALTER TABLE knowledge_documents ADD COLUMN {col_name} {col_type}"
                ))
                logger.info("Migrated knowledge_documents: added column %s", col_name)
        conn.commit()
