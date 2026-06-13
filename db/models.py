"""SQLAlchemy ORM models."""
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Text, JSON, ForeignKey
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Environment(Base):
    __tablename__ = "environments"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    env_type = Column(String, nullable=False)  # e.g., dev, test, staging
    region = Column(String, nullable=True)
    components = Column(JSON, nullable=True)  # list of component names
    host = Column(String, nullable=True)
    port = Column(Integer, nullable=True)
    status = Column(String, default="unknown")  # available, busy, unknown
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Integer, default=1)

    validations = relationship("ValidationRecord", back_populates="environment",
                               cascade="all, delete-orphan")


class ValidationRecord(Base):
    __tablename__ = "validation_records"

    id = Column(Integer, primary_key=True)
    environment_id = Column(Integer, ForeignKey("environments.id"), nullable=False)
    session_id = Column(String, nullable=True)
    probe_result = Column(JSON, nullable=True)
    matched_components = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    environment = relationship("Environment", back_populates="validations")


class KnowledgeDocument(Base):
    __tablename__ = "knowledge_documents"

    id = Column(Integer, primary_key=True)
    doc_id = Column(String, unique=True, nullable=False)
    title = Column(String, nullable=True)
    content = Column(Text, nullable=False)
    category = Column(String, nullable=False)  # sdk, manual, spec
    source = Column(String, nullable=True)
    metadata_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Integer, default=1)


class UserSession(Base):
    __tablename__ = "user_sessions"

    id = Column(Integer, primary_key=True)
    session_id = Column(String, unique=True, nullable=False)
    extracted_fields = Column(JSON, default=dict)
    history = Column(JSON, default=list)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
