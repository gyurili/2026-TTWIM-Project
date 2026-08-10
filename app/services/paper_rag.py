"""러닝 RAG 챗봇이 참조하는 논문 청크(paper_chunks) 검색 모듈.

1단계 pgvector 코사인 검색으로 후보를 넓게 뽑고(top-`candidates`), 2단계로 LLM 리랭킹을 거쳐
질문에 실제로 답이 되는 상위 `top_k`개만 반환한다. 리랭킹 효과는 scripts/eval_retrieval.py로
실측하며, 개선이 없으면 _RERANK_ENABLED를 False로 내려 지연을 절약한다.

인덱스 적재는 scripts/ingest_papers.py가 오프라인으로 수행하며, 이 모듈은 읽기만 한다.
"""

import logging

from google import genai
from google.genai import types
from langsmith import traceable
from pydantic import BaseModel, Field
from sqlalchemy import text as sa_text
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.session import SessionLocal

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIM = 768
RERANK_MODEL = "gemini-3.1-flash-lite"

_DEFAULT_TOP_K = 5
_DEFAULT_CANDIDATES = 20
_MAX_PER_DOC = 3  # 한 논문이 후보 상위권을 독점하지 않도록 논문당 최대 채택 청크 수
_MIN_SCORE = 0.55  # 코사인 유사도 하한. 미달이면 "근거 없음"으로 처리해 환각을 막는다

_RERANK_ENABLED = True  # scripts/eval_retrieval.py 결과로 확정. 개선 없으면 False로 내려 지연 절약


def load_index() -> None:
    """서버 기동 시 호출하는 가벼운 헬스체크. pgvector는 상태를 메모리에 올리지 않으므로
    실제로 하는 일은 paper_chunks에 데이터가 있는지 확인해 인제스천 누락을 조기에 경고하는 것뿐이다."""
    db = SessionLocal()
    try:
        count = db.execute(sa_text("SELECT count(*) FROM paper_chunks")).scalar()
        if not count:
            logger.warning(
                "paper_chunks가 비어 있습니다. scripts/ingest_papers.py를 실행해 논문을 적재하세요. "
                "RAG 챗봇은 논문 근거 없이 동작합니다."
            )
        else:
            logger.info("paper_chunks 로드 확인: %d개 청크", count)
    except Exception as e:
        logger.warning("paper_chunks 상태 확인 실패 (테이블 미생성 가능): %s", e)
    finally:
        db.close()


def _embed_query(client: genai.Client, query: str) -> list[float]:
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=[query],
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=EMBEDDING_DIM,
        ),
    )
    return response.embeddings[0].values


def _vector_search(db: Session, query_vector: list[float], candidates: int, lang: str | None) -> list[dict]:
    # ef_search를 기본값(40)보다 높여 HNSW 근사 검색의 리콜을 확보한다. 트랜잭션 범위 내에서만 적용.
    db.execute(sa_text("SET LOCAL hnsw.ef_search = 64"))
    lang_filter = "AND lang = :lang" if lang else ""
    rows = db.execute(
        sa_text(
            f"""
            SELECT chunk_id, doc_id, title, page, lang, content,
                   1 - (embedding <=> :qvec) AS score
            FROM paper_chunks
            WHERE 1=1 {lang_filter}
            ORDER BY embedding <=> :qvec
            LIMIT :limit
            """
        ),
        {"qvec": str(query_vector), "limit": candidates, **({"lang": lang} if lang else {})},
    ).mappings().all()
    return [dict(r) for r in rows]


def _diversify(chunks: list[dict], max_per_doc: int = _MAX_PER_DOC) -> list[dict]:
    per_doc_count: dict[str, int] = {}
    out = []
    for c in chunks:
        n = per_doc_count.get(c["doc_id"], 0)
        if n >= max_per_doc:
            continue
        per_doc_count[c["doc_id"]] = n + 1
        out.append(c)
    return out


class _RerankOutput(BaseModel):
    ranked_indices: list[int] = Field(description="후보 번호를 질문과의 관련도 높은 순서로 정렬한 리스트")


@traceable(name="rerank_paper_chunks")
def _rerank(client: genai.Client, query: str, candidates: list[dict], top_k: int) -> list[dict]:
    numbered = "\n".join(
        f"[{i}] ({c['title']} p.{c['page']}) {c['content'][:300]}" for i, c in enumerate(candidates)
    )
    prompt = (
        f"[사용자 질문]\n{query}\n\n"
        f"[후보 문헌 발췌]\n{numbered}\n\n"
        "위 후보들을 사용자 질문에 실제로 답이 되는 정도가 높은 순서로 번호만 정렬해 반환하세요. "
        "질문과 무관한 후보는 목록에서 제외해도 됩니다."
    )
    try:
        response = client.models.generate_content(
            model=RERANK_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=500,
                response_mime_type="application/json",
                response_schema=_RerankOutput,
                http_options=types.HttpOptions(timeout=15_000),
            ),
        )
        parsed = getattr(response, "parsed", None)
        if parsed is None and response.text:
            parsed = _RerankOutput.model_validate_json(response.text)
        if parsed is None:
            raise ValueError("리랭킹 응답 파싱 실패")
        ranked = [candidates[i] for i in parsed.ranked_indices if 0 <= i < len(candidates)]
        return ranked[:top_k] if ranked else candidates[:top_k]
    except Exception as e:
        logger.warning("논문 리랭킹 실패, 벡터 점수 순으로 폴백: %s", e)
        return candidates[:top_k]


@traceable(name="search_paper_chunks")
def search(
    db: Session,
    query: str,
    top_k: int = _DEFAULT_TOP_K,
    candidates: int = _DEFAULT_CANDIDATES,
    lang: str | None = None,
) -> list[dict]:
    """질문에 대한 논문 근거를 검색한다. 결과가 비면 호출부가 "근거 없음"으로 처리해야 한다."""
    client = genai.Client(api_key=settings.GEMINI_API_KEY)

    try:
        query_vector = _embed_query(client, query)
    except Exception as e:
        logger.error("검색 쿼리 임베딩 실패: %s", e)
        return []

    try:
        rows = _vector_search(db, query_vector, candidates, lang)
    except Exception as e:
        logger.error("pgvector 검색 실패: %s", e)
        return []

    rows = [r for r in rows if r["score"] >= _MIN_SCORE]
    if not rows:
        return []

    rows = _diversify(rows)

    if _RERANK_ENABLED and len(rows) > top_k:
        rows = _rerank(client, query, rows, top_k)
    else:
        rows = rows[:top_k]

    return rows
