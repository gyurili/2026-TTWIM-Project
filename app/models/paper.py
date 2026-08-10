from pgvector.sqlalchemy import Vector
from sqlalchemy import BigInteger, Column, DateTime, Index, Integer, String, Text

from app.core.time import now_kst_naive
from app.db.base import Base

EMBEDDING_DIM = 768


class PaperChunk(Base):
    """RAG 챗봇이 검색할 논문 청크. app/data/paper/*.pdf를 scripts/ingest_papers.py로 추출·청킹·임베딩해 적재한다"""

    __tablename__ = "paper_chunks"

    chunk_id   = Column(BigInteger, primary_key=True, autoincrement=True)
    doc_id     = Column(String(200), nullable=False, index=True)  # 논문 파일 stem
    title      = Column(String(300), nullable=False)
    page       = Column(Integer, nullable=False)
    lang       = Column(String(2), nullable=False)  # "ko" | "en"
    content    = Column(Text, nullable=False)
    embedding  = Column(Vector(EMBEDDING_DIM), nullable=False)
    created_at = Column(DateTime, default=now_kst_naive)

    __table_args__ = (
        Index(
            "ix_paper_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
            postgresql_with={"m": 16, "ef_construction": 64},
        ),
    )
