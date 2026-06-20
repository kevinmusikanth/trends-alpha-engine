from functools import lru_cache
from os import getenv

from dotenv import load_dotenv
from pydantic import BaseModel


class Settings(BaseModel):
    database_url: str = "postgresql+psycopg2://tae:tae@localhost:5432/tae"
    sec_user_agent: str = "TrendsAlphaEngine/0.1 contact@example.com"
    default_universe: str = "sp500"


@lru_cache
def get_settings() -> Settings:
    load_dotenv()
    return Settings(
        database_url=getenv("TAE_DATABASE_URL", Settings().database_url),
        sec_user_agent=getenv("TAE_SEC_USER_AGENT", Settings().sec_user_agent),
        default_universe=getenv("TAE_DEFAULT_UNIVERSE", Settings().default_universe),
    )

