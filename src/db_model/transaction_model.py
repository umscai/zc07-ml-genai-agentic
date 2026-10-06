from datetime import datetime, timezone
from sqlalchemy import Column, Integer, Float, String, Boolean, DateTime
from sqlalchemy.orm import declarative_base
from config import settings
import utils as ut
import logging
logger = logging.getLogger(__name__)

Base = declarative_base()

class TransLog(Base):
    __tablename__ = "transaction_logs"

    index = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    V1 = Column(Float, nullable=False)
    V2 = Column(Float, nullable=False)
    V3 = Column(Float, nullable=False)
    V4 = Column(Float, nullable=False)
    V5 = Column(Float, nullable=False)
    V6 = Column(Float, nullable=False)
    V7 = Column(Float, nullable=False)
    V8 = Column(Float, nullable=False)
    V9 = Column(Float, nullable=False)
    V10 = Column(Float, nullable=False)
    V11 = Column(Float, nullable=False)
    V12 = Column(Float, nullable=False)
    V13 = Column(Float, nullable=False)
    V14 = Column(Float, nullable=False)
    V15 = Column(Float, nullable=False)
    V16 = Column(Float, nullable=False)
    V17 = Column(Float, nullable=False)
    V18 = Column(Float, nullable=False)
    V19 = Column(Float, nullable=False)
    V20 = Column(Float, nullable=False)
    V21 = Column(Float, nullable=False)
    V22 = Column(Float, nullable=False)
    V23 = Column(Float, nullable=False)
    V24 = Column(Float, nullable=False)
    V25 = Column(Float, nullable=False)
    V26 = Column(Float, nullable=False)
    V27 = Column(Float, nullable=False)
    V28 = Column(Float, nullable=False)
    Amount = Column(Float, nullable=False)
    HourT = Column(Integer, nullable=False)
    actual_class = Column(Integer, nullable=False)
    pred_class = Column(Integer, nullable=False)



def init_trans_db(force_reset:bool = False):
    """
    init and Create a new TRANSACTION SQLite database
    Args:
        - force_reset: if True, reset existing dB and then init/setup, otherwise no action is taken
    Returns:
        None
    Raises:
        Exception: Re-raises any exception caught during processing after logging.
    """
    try:
        ut.init_and_setup_db(settings.TRANSACTION_DIR / settings.TRANSACTION_DB, Base, force_reset)
    except Exception as e:
        logger.error(f"Error in init_trans_db: {e}")
        raise


if __name__ == "__main__":
    init_trans_db(force_reset=True)
