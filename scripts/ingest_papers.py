"""app/data/paper/*.pdf를 청킹·임베딩해 RDS paper_chunks 테이블에 적재하는 오프라인 스크립트.

서버 런타임에서는 import되지 않는다 (사람이 필요할 때 수동 실행).

사용법:
    python scripts/ingest_papers.py --dry-run          # PDF별 추출 문자 수만 확인
    python scripts/ingest_papers.py                    # 전체 적재
    python scripts/ingest_papers.py --only "Effect of stride length on the running.pdf"

적재는 doc_id(파일 stem) 단위로 DELETE 후 INSERT라 재실행해도 중복이 쌓이지 않는다.
"""

from __future__ import annotations

import argparse
import io
import logging
import random
import re
import sys
import time
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    # Windows 콘솔 기본 코드페이지(cp949)는 논문 파일명의 일부 유니코드 문자(예: U+2011)를
    # 인코딩하지 못해 dry-run 출력이 죽는다. 표준출력을 UTF-8로 강제한다.
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from google import genai
from google.genai import types
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from sqlalchemy import text as sa_text

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 스크립트 단독 실행 시 app.* import 허용

from app.core.config import settings  # noqa: E402
from app.db.session import SessionLocal, engine  # noqa: E402
from app.db.base import Base  # noqa: E402
import app.models.paper  # noqa: E402,F401  (Base.metadata에 등록하기 위해 import만 필요)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ingest_papers")

PAPER_DIR = Path(__file__).resolve().parents[1] / "app" / "data" / "paper"
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIM = 768
EMBED_BATCH_SIZE = 100

CHUNK_SIZE = 900
CHUNK_OVERLAP = 140
MIN_CHUNK_CHARS = 100

# feedback.py::_is_retryable_error, _call_llm과 동일한 재시도 정책 (429/5xx/timeout만 재시도)
RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}
MAX_RETRIES = 4
RETRY_BACKOFF_BASE_SEC = 2.0

_REFERENCES_HEADING = re.compile(r"\n\s*(references|bibliography|참고\s*문헌)\s*\n", re.IGNORECASE)
_HANGUL = re.compile(r"[가-힣]")


def _is_retryable_error(e: Exception) -> bool:
    status = getattr(e, "code", None) or getattr(getattr(e, "response", None), "status_code", None)
    if status is not None:
        return status in RETRYABLE_STATUS_CODES
    name = type(e).__name__.lower()
    return any(keyword in name for keyword in ("timeout", "connection", "deadline"))


def extract_pages(pdf_path: Path) -> list[str]:
    """페이지별 텍스트 리스트를 반환한다. 페이지 인덱스는 1-based로 다룬다 (호출부에서 enumerate(1,...)).
    일부 PDF는 pypdf 추출 시 NUL(0x00) 문자가 섞여 나오는데, Postgres TEXT 컬럼은 이를 저장할 수 없어
    INSERT가 실패한다 (ValueError: A string literal cannot contain NUL characters). 여기서 제거한다."""
    reader = PdfReader(str(pdf_path))
    return [(page.extract_text() or "").replace("\x00", "") for page in reader.pages]


def strip_references(pages: list[str]) -> list[str]:
    """References/Bibliography 절 이후는 인용 목록이라 RAG 근거로 부적합하므로 잘라낸다.
    표제가 나온 페이지부터 그 페이지의 표제 이전 텍스트만 남기고 이후 페이지는 버린다."""
    out: list[str] = []
    for page_text in pages:
        m = _REFERENCES_HEADING.search("\n" + page_text)
        if m:
            out.append(page_text[: m.start()])
            break
        out.append(page_text)
    return out


def detect_lang(text: str) -> str:
    hangul_count = len(_HANGUL.findall(text))
    ratio = hangul_count / max(len(text), 1)
    return "ko" if ratio > 0.15 else "en"


def chunk_pdf(pdf_path: Path) -> list[dict]:
    """PDF 한 편을 {doc_id, title, page, lang, content} 청크 리스트로 변환한다."""
    doc_id = pdf_path.stem
    title = doc_id
    pages = strip_references(extract_pages(pdf_path))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[dict] = []
    for page_num, page_text in enumerate(pages, start=1):
        if not page_text.strip():
            continue
        for piece in splitter.split_text(page_text):
            piece = piece.strip()
            if len(piece) < MIN_CHUNK_CHARS:
                continue
            chunks.append(
                {
                    "doc_id": doc_id,
                    "title": title,
                    "page": page_num,
                    "lang": detect_lang(piece),
                    "content": piece,
                }
            )
    return chunks


def embed_batch(client: genai.Client, texts: list[str]) -> list[list[float]]:
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=texts,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=EMBEDDING_DIM,
                ),
            )
            return [e.values for e in response.embeddings]
        except Exception as e:
            last_error = e
            retryable = _is_retryable_error(e)
            logger.warning("임베딩 호출 실패 (시도 %d/%d, retryable=%s): %s", attempt + 1, MAX_RETRIES + 1, retryable, e)
            if not retryable or attempt == MAX_RETRIES:
                break
            time.sleep(RETRY_BACKOFF_BASE_SEC * (2**attempt) + random.uniform(0, 0.5))
    raise RuntimeError(f"임베딩 호출 최종 실패: {last_error}") from last_error


def embed_chunks(client: genai.Client, chunks: list[dict]) -> None:
    for i in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch = chunks[i : i + EMBED_BATCH_SIZE]
        vectors = embed_batch(client, [c["content"] for c in batch])
        for chunk, vector in zip(batch, vectors):
            chunk["embedding"] = vector


def upsert_doc(db, doc_id: str, chunks: list[dict]) -> None:
    """doc_id 단위로 기존 청크를 지우고 새로 넣는다 (멱등한 재적재)."""
    db.execute(sa_text("DELETE FROM paper_chunks WHERE doc_id = :doc_id"), {"doc_id": doc_id})
    if chunks:
        db.execute(
            sa_text(
                """
                INSERT INTO paper_chunks (doc_id, title, page, lang, content, embedding)
                VALUES (:doc_id, :title, :page, :lang, :content, :embedding)
                """
            ),
            [
                {
                    "doc_id": c["doc_id"],
                    "title": c["title"],
                    "page": c["page"],
                    "lang": c["lang"],
                    "content": c["content"],
                    "embedding": str(c["embedding"]),  # pgvector는 '[0.1,0.2,...]' 텍스트 표현을 받는다
                }
                for c in chunks
            ],
        )
    db.commit()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="PDF별 추출 문자 수만 출력하고 종료")
    parser.add_argument("--only", type=str, default=None, help="특정 PDF 파일명만 처리")
    parser.add_argument(
        "--force", action="store_true",
        help="이미 paper_chunks에 적재된 doc_id도 재적재. 기본은 이미 적재된 문서를 건너뛴다 (중단 후 재실행 대비).",
    )
    args = parser.parse_args()

    pdf_paths = sorted(PAPER_DIR.glob("*.pdf"))
    if args.only:
        pdf_paths = [p for p in pdf_paths if p.name == args.only]
        if not pdf_paths:
            logger.error("--only에 해당하는 파일을 찾을 수 없습니다: %s", args.only)
            sys.exit(1)

    if not pdf_paths:
        logger.error("PDF를 찾을 수 없습니다: %s", PAPER_DIR)
        sys.exit(1)

    if args.dry_run:
        print(f"{'파일명':<70} {'페이지':>6} {'문자수':>10} {'문자/페이지':>10}")
        for pdf_path in pdf_paths:
            pages = extract_pages(pdf_path)
            total_chars = sum(len(p) for p in pages)
            per_page = total_chars / max(len(pages), 1)
            flag = "  ⚠ 스캔본 의심 (OCR 필요)" if per_page < 200 else ""
            print(f"{pdf_path.name:<70} {len(pages):>6} {total_chars:>10} {per_page:>10.0f}{flag}")
        return

    Base.metadata.create_all(bind=engine, tables=[app.models.paper.PaperChunk.__table__])

    client = genai.Client(api_key=settings.GEMINI_API_KEY)
    db = SessionLocal()
    total_chunks = 0
    try:
        existing_doc_ids: set[str] = set()
        if not args.force:
            existing_doc_ids = {
                row[0] for row in db.execute(sa_text("SELECT DISTINCT doc_id FROM paper_chunks")).all()
            }

        for pdf_path in pdf_paths:
            if pdf_path.stem in existing_doc_ids:
                logger.info("건너뜀 (이미 적재됨, --force로 재적재 가능): %s", pdf_path.name)
                continue
            logger.info("처리 중: %s", pdf_path.name)
            chunks = chunk_pdf(pdf_path)
            if not chunks:
                logger.warning("추출된 청크가 없습니다 (스캔 이미지 PDF일 수 있음): %s", pdf_path.name)
                continue
            embed_chunks(client, chunks)
            upsert_doc(db, pdf_path.stem, chunks)
            total_chunks += len(chunks)
            logger.info("적재 완료: %s (%d 청크)", pdf_path.name, len(chunks))
    finally:
        db.close()

    logger.info("전체 완료: %d편, %d 청크", len(pdf_paths), total_chunks)


if __name__ == "__main__":
    main()
