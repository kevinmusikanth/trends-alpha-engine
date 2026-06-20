from tae.db.models import Base
from tae.db.session import build_engine


def main() -> None:
    engine = build_engine()
    Base.metadata.create_all(engine)
    print("TAE database tables initialized.")


if __name__ == "__main__":
    main()

