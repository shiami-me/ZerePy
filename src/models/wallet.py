from datetime import datetime
from sqlalchemy import Column, String, DateTime, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from src.database import Base


class UserWallet(Base):
    __tablename__ = "user_wallets"
    
    id = Column(String, primary_key=True)
    user_address = Column(String, index=True)
    wallet_address = Column(String, unique=True, index=True)
    wallet_id = Column(String, unique=True)
    chain_type = Column(String, default="ethereum")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_accessed_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # JWT token for the user's session
    session_token = Column(String)
    session_expires_at = Column(DateTime)
    
    def __repr__(self):
        return f"<UserWallet {self.wallet_address}>"
