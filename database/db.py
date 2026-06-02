from sqlalchemy import create_engine, Column, String, Float, Integer, DateTime, Text, JSON
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

DATABASE_URL = "sqlite:///./circulardrive.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class PassportRecord(Base):
    __tablename__ = "passports"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    component_id = Column(String, unique=True, index=True)
    battery_id = Column(String)
    vehicle_id = Column(String)
    manufacturer = Column(String)
    model = Column(String)
    chemistry = Column(String)
    rated_capacity_kwh = Column(Float)
    voltage = Column(Float)
    module_count = Column(Integer)
    cell_count = Column(Integer)
    predicted_soh = Column(Float)
    rul_cycles = Column(Integer)
    confidence_score = Column(String)
    grade = Column(String)
    recommendation = Column(String)
    risk_flags = Column(Text)  # JSON string
    materials = Column(Text)  # JSON string
    manufacturing_date = Column(String)
    service_history = Column(Text)  # JSON string
    embodied_carbon_kg = Column(Float)
    recycled_content_pct = Column(Float)
    recovery_score = Column(Float)
    disassembly_notes = Column(Text)
    safety_warnings = Column(Text)
    qr_code_path = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    component_id = Column(String, index=True)
    input_data = Column(Text)
    predicted_soh = Column(Float)
    grade = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)


class VehicleRecord(Base):
    __tablename__ = "vehicles"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String, index=True)
    type = Column(String)
    year = Column(String)
    msrp = Column(String)
    data = Column(JSON)  # Store the full JSON payload
    created_at = Column(DateTime, default=datetime.utcnow)


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

