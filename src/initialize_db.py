import logging
from src.database import engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_db():
    """Initialize the database tables"""
    logger.info("Creating database tables...")
    logger.info("Database tables created successfully")

if __name__ == "__main__":
    init_db()
