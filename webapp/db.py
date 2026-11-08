"""Task 5 - database trail. Every prediction becomes one row: what was decided, when, by which model,
how confidently. Works with PostgreSQL (docker compose) or SQLite (local dev / tests)."""
import time
from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Integer, String, create_engine, select, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    __tablename__ = "predictions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True,
                                                 default=lambda: datetime.now(timezone.utc))
    filename: Mapped[str] = mapped_column(String(120))
    label: Mapped[str] = mapped_column(String(20))
    confidence: Mapped[float] = mapped_column(Float)
    prob_ai: Mapped[float] = mapped_column(Float)
    threshold: Mapped[float] = mapped_column(Float)
    model_name: Mapped[str] = mapped_column(String(120))
    model_version: Mapped[str] = mapped_column(String(40))
    inference_mode: Mapped[str] = mapped_column(String(10))   # local | remote
    latency_ms: Mapped[float] = mapped_column(Float)
    image_format: Mapped[str] = mapped_column(String(10))
    image_width: Mapped[int] = mapped_column(Integer)
    image_height: Mapped[int] = mapped_column(Integer)
    file_size_bytes: Mapped[int] = mapped_column(Integer)
    client_ip: Mapped[str] = mapped_column(String(64), default="")


class Database:
    def __init__(self, url: str, retries: int = 15, delay: float = 2.0):
        self.engine = create_engine(url, pool_pre_ping=True, future=True)
        for attempt in range(1, retries + 1):        # extra safety on top of compose health checks
            try:
                with self.engine.connect() as c:
                    c.execute(text("SELECT 1"))
                break
            except OperationalError as e:
                if attempt == retries:
                    raise
                print(f"[db] not ready ({e.__class__.__name__}), retry {attempt}/{retries}", flush=True)
                time.sleep(delay)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(self.engine, expire_on_commit=False)

    def add(self, **fields) -> Prediction:
        with self.Session.begin() as s:
            row = Prediction(**fields)
            s.add(row)
        return row

    def recent(self, limit: int = 100) -> list[Prediction]:
        with self.Session() as s:
            return list(s.scalars(select(Prediction).order_by(Prediction.id.desc()).limit(limit)))

    def ping(self) -> bool:
        try:
            with self.engine.connect() as c:
                c.execute(text("SELECT 1"))
            return True
        except Exception:  # noqa: BLE001
            return False
