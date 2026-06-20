from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from tae.config import get_settings


def build_engine():
    return create_engine(get_settings().database_url, future=True)


SessionLocal = sessionmaker(bind=build_engine(), autoflush=False, autocommit=False, future=True)

