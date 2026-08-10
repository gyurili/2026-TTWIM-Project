# Azas팀 FastAPI 레포

아자쓰 팀의 FastAPI 저장소입니다.

## 실행 방법

1. 가상환경 생성: `python -m venv venv`
2. 가상환경 실행: `source venv/Scripts/activate`
3. 패키지 설치: `pip install -r requirements.txt`
4. 서버 시작: `uvicorn app.main:app --reload`

## 러닝 리포트 RAG 챗봇

`POST/GET/DELETE /running/{session_id}/chat` — 최종 러닝 리포트 화면에서 사용자가 후속 질문을 하면
`app/data/paper/`의 논문 20편(영/한 혼재)을 근거로 Gemini가 한국어로 답한다.

- **인덱스**: RDS Postgres + pgvector(HNSW). 논문은 정적 데이터라 RDS와 별개 벡터DB를 두지 않고,
  이미 떠 있는 RDS에 `paper_chunks` 테이블로 적재한다 (`scripts/ingest_papers.py`).
- **검색**: pgvector 코사인 top-20 → LLM(`gemini-3.1-flash-lite`) 리랭킹 top-5.
- **멀티턴**: 대화 이력이 있으면 검색 전 질문을 독립 질문으로 재작성한다 (`app/services/chat.py::_rewrite_query`).

### 검색 품질 실측 (`scripts/eval_retrieval.py`, `app/data/eval_questions.json`)

리랭킹·질문 재작성 둘 다 감이 아니라 아래 수치로 유지 여부를 결정했다 (둘 다 유지 확정):

**단일 질문 20개 — 리랭킹 효과** (Hit@5는 이미 포화, 순위 지표에서 개선이 드러남)

| 설정 | Hit@5 | MRR | mean rank |
|---|---|---|---|
| rerank 없음 | 1.0 | 0.864 | 1.5 |
| rerank 적용 | 1.0 | **0.96** | **1.2** |

**멀티턴 8개 중 주제 전환 케이스 4개** — 재작성이 없으면 검색이 실제로 망가지는 걸 보여주는 핵심 지표

| 설정 | Hit@5 | MRR | mean rank |
|---|---|---|---|
| 재작성 없음, rerank 없음 | 0.5 | 0.583 | 3.5 |
| 재작성 없음, rerank 적용 | 0.5 | 0.583 | 3.5 |
| 재작성 적용, rerank 없음 | 1.0 | 0.6 | 3.0 |
| 재작성 적용, rerank 적용 | **1.0** | **0.8** | **2.0** |

재작성 없이는 주제가 바뀐 질문의 절반이 직전 질문의 용어에 끌려가 엉뚱한 논문을 검색한다.
재작성을 켜면 이 문제가 해소되고, 리랭킹을 더하면 정답 문서의 순위가 한 번 더 올라간다.

### 재적재 / 재평가

```bash
python scripts/ingest_papers.py --dry-run   # PDF별 추출 문자 수 확인 (스캔본 판별)
python scripts/ingest_papers.py             # 적재 (doc_id 단위 멱등, 이미 적재된 문서는 건너뜀)
python scripts/ingest_papers.py --force     # 전체 재적재
python scripts/eval_retrieval.py            # 검색 품질 재측정
```
