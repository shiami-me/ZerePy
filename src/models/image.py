from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class GeneratedImage(Base):
    __tablename__ = 'generated_images'
    
    id = Column(Integer, primary_key=True)
    filename = Column(String, nullable=False)
    ipfs_hash=Column(String, nullable=False)
    prompt = Column(String, nullable=False)
    model = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
