import logging
from src.database import engine
import src.models.wallet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_db():
    """Initialize the database tables"""
    logger.info("Creating database tables...")
    src.models.wallet.Base.metadata.create_all(bind=engine)
    logger.info("Database tables created successfully")

if __name__ == "__main__":
    init_db()
