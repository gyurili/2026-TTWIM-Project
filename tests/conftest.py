import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.db.base import Base
from app.db.session import get_db

# 유저 도메인 테스트용 별도 DB. dev/RDS와 같은 서버지만 dbname만 ajas_test로 분리해서
# 실제 데이터와 절대 섞이지 않게 한다.
TEST_DATABASE_URL = (
    f"postgresql://{settings.DB_USER}:{settings.DB_PASSWORD}"
    f"@{settings.DB_HOST}:{settings.DB_PORT}/ajas_test"
)

engine = create_engine(TEST_DATABASE_URL)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(scope="session", autouse=True)
def _setup_test_db():
    # app.main을 import해야 라우터를 거쳐 모든 모델(User 포함)이 Base.metadata에 등록된다.
    from app.main import app  # noqa: F401

    Base.metadata.create_all(bind=engine)
    yield


@pytest.fixture()
def db_session():
    """테스트 하나당 트랜잭션 하나. 끝나면 무조건 롤백해서 DB에 아무 흔적도 안 남긴다.

    app 코드(repositories)가 내부적으로 db.commit()을 직접 호출하기 때문에, 바깥
    트랜잭션 위에서 SAVEPOINT(nested transaction)를 써야 한다 — 그냥 begin()만 쓰면
    app의 commit()이 바깥 트랜잭션까지 끝내버려서 마지막 rollback이 무의미해진다.
    """
    connection = engine.connect()
    transaction = connection.begin()
    session = TestingSessionLocal(bind=connection)
    session.begin_nested()

    @event.listens_for(session, "after_transaction_end")
    def _restart_savepoint(sess, trans):
        if trans.nested and not trans._parent.nested:
            sess.begin_nested()

    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()


@pytest.fixture()
def client(db_session):
    from app.main import app

    def _override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = _override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
