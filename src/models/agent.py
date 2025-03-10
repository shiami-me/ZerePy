from sqlalchemy import Column, Integer, String, DateTime, Boolean, JSON
from sqlalchemy.sql import func
from ..database import Base
import uuid

class Agent(Base):
    __tablename__ = "agents"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_address = Column(String, nullable=False, index=True)
    name = Column(String, nullable=False)
    is_one_time = Column(Boolean, default=False)
    agents_list = Column(JSON, nullable=False)
    prompts = Column(JSON, nullable=False)
    data = Column(JSON, nullable=False)
    logs = Column(JSON, default=lambda: [])
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    
    def add_log(self, log):
        """Add a log entry to this agent"""
        if not self.logs:
            self.logs = []
        print("log", log)
        current_logs = self.logs.copy() if self.logs else []
        current_logs.append(log)
        # Force SQLAlchemy to detect the change by reassigning the entire attribute
        self.logs = current_logs