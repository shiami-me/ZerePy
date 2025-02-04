from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
import os
from dotenv import load_dotenv

load_dotenv()

POSTGRES_DB_URI = os.getenv("POSTGRES_DB_URI")
if not POSTGRES_DB_URI:
    raise ValueError("POSTGRES_DB_URI environment variable is not set")

engine = create_engine(POSTGRES_DB_URI)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
