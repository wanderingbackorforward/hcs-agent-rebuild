"""Session manager."""
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from config.database import db_config


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
