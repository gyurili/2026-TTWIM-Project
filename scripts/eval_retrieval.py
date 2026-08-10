"""app/data/eval_questions.json의 골든 질문으로 논문 검색 품질을 측정하는 스크립트.

목적: app/services/paper_rag.py의 _RERANK_ENABLED와 app/services/chat.py의 쿼리 재작성이
실제로 검색 정확도를 올리는지 감이 아니라 숫자로 확인한다 (계획 §7).

지표:
- Hit@5      : top-5 안에 정답 문서가 하나라도 있는 비율
- MRR        : 정답 문서 중 가장 순위가 높은 것의 역순위 평균 (1/rank)
- mean rank  : 정답 문서 중 가장 순위가 높은 것의 평균 순위 (낮을수록 좋음, 후보 풀 밖이면 penalty)

단일 질문 세트는 rerank on/off 2가지, 멀티턴 세트는 rewrite x rerank 4가지 설정을 비교한다.
멀티턴 없는 단일 질문에 rewrite를 적용하는 건 의미가 없어(재작성할 대화 이력이 없음) 대상에서 뺐다.

사용법:
    python scripts/eval_retrieval.py
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from google import genai  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.db.session import SessionLocal  # noqa: E402
from app.services import paper_rag  # noqa: E402
from app.services.chat import _rewrite_query  # noqa: E402

EVAL_PATH = Path(__file__).resolve().parents[1] / "app" / "data" / "eval_questions.json"
CANDIDATES = 20  # paper_rag.search의 1단계 벡터 검색 폭 (프로덕션 기본값과 동일)
TOP_K = 5  # 챗봇이 실제로 프롬프트에 넣는 최종 개수 (프로덕션 기본값과 동일)
NOT_FOUND_RANK = TOP_K + 1  # 최종 top-K 밖으로 밀려난 경우의 페널티 순위


def _best_rank(ranked_doc_ids: list[str], expected_doc_ids: list[str]) -> int:
    """ranked_doc_ids(1위부터) 중 expected에 속하는 첫 항목의 1-based 순위. 없으면 NOT_FOUND_RANK."""
    for i, doc_id in enumerate(ranked_doc_ids, start=1):
        if doc_id in expected_doc_ids:
            return i
    return NOT_FOUND_RANK


def _score(ranks: list[int], hit_at: int = TOP_K) -> dict:
    n = len(ranks)
    hit = sum(1 for r in ranks if r <= hit_at) / n
    mrr = sum(1 / r for r in ranks) / n
    mean_rank = sum(ranks) / n
    return {"n": n, "hit@5": round(hit, 3), "mrr": round(mrr, 3), "mean_rank": round(mean_rank, 2)}


def _search_ranked_doc_ids(db, query: str, top_k: int, rerank: bool) -> list[str]:
    """paper_rag.search와 같은 파이프라인(임베딩 → 벡터 top-N → 다양성 필터 → [리랭킹])을 쓰되,
    _RERANK_ENABLED 모듈 상수를 매 호출 건드리지 않도록 순위 비교에 필요한 만큼만 재구성한다."""
    original = paper_rag._RERANK_ENABLED
    paper_rag._RERANK_ENABLED = rerank
    try:
        results = paper_rag.search(db, query, top_k=top_k, candidates=CANDIDATES)
    finally:
        paper_rag._RERANK_ENABLED = original
    return [r["doc_id"] for r in results]


def eval_candidate_pool_recall(db, questions: list[dict]) -> dict:
    """진단용: 리랭킹 이전, 임베딩+다양성 필터 단계만으로 top-CANDIDATES 안에 정답이 있는지.
    이게 낮으면 리랭킹을 아무리 잘해도 못 살리는 병목이 임베딩/청킹 쪽에 있다는 뜻이다."""
    ranks = []
    for q in questions:
        ranked = _search_ranked_doc_ids(db, q["question"], top_k=CANDIDATES, rerank=False)
        rank = next((i for i, d in enumerate(ranked, start=1) if d in q["expected_doc_ids"]), CANDIDATES + 1)
        ranks.append(rank)
    return {"candidates=20 (rerank 없음)": _score(ranks, hit_at=CANDIDATES)}


def eval_single_turn(db, questions: list[dict]) -> dict:
    out = {}
    for rerank in (False, True):
        ranks = []
        for q in questions:
            ranked = _search_ranked_doc_ids(db, q["question"], top_k=TOP_K, rerank=rerank)
            ranks.append(_best_rank(ranked, q["expected_doc_ids"]))
        out[f"rerank={rerank}"] = _score(ranks)
    return out


def eval_multi_turn(db, client, questions: list[dict]) -> dict:
    out = {}
    for rewrite in (False, True):
        for rerank in (False, True):
            ranks = []
            for q in questions:
                history = [SimpleNamespace(role=h["role"], content=h["content"]) for h in q["history"]]
                if rewrite:
                    query = _rewrite_query(client, q["question"], history)
                else:
                    last_user = next((h["content"] for h in reversed(q["history"]) if h["role"] == "user"), "")
                    query = f"{last_user} {q['question']}".strip()
                ranked = _search_ranked_doc_ids(db, query, top_k=TOP_K, rerank=rerank)
                ranks.append(_best_rank(ranked, q["expected_doc_ids"]))
            out[f"rewrite={rewrite},rerank={rerank}"] = _score(ranks)
    return out


def eval_multi_turn_topic_shift_only(db, client, questions: list[dict]) -> dict:
    shift_questions = [q for q in questions if q.get("topic_shift")]
    return eval_multi_turn(db, client, shift_questions)


def _print_table(title: str, results: dict) -> None:
    print(f"\n=== {title} ===")
    header = f"{'설정':<28} {'n':>4} {'Hit@5':>8} {'MRR':>8} {'mean_rank':>10}"
    print(header)
    print("-" * len(header))
    for label, s in results.items():
        print(f"{label:<28} {s['n']:>4} {s['hit@5']:>8} {s['mrr']:>8} {s['mean_rank']:>10}")


def main() -> None:
    data = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    single = data["single_turn"]
    multi = data["multi_turn"]

    client = genai.Client(api_key=settings.GEMINI_API_KEY)
    db = SessionLocal()
    try:
        pool_results = eval_candidate_pool_recall(db, single + [
            {**q, "question": q["history"][-1]["content"] + " " + q["question"]} for q in multi
        ])
        single_results = eval_single_turn(db, single)
        multi_results = eval_multi_turn(db, client, multi)
        shift_results = eval_multi_turn_topic_shift_only(db, client, multi)
    finally:
        db.close()

    _print_table("진단: 후보 풀(top-20) 재현율 — 임베딩/청킹 단계 자체의 병목 확인", pool_results)
    _print_table(f"단일 질문 {len(single)}개 — 리랭킹 효과", single_results)
    _print_table(f"멀티턴 질문 {len(multi)}개 — 재작성/리랭킹 효과", multi_results)
    _print_table(
        f"멀티턴 중 주제 전환 케이스 {len([q for q in multi if q.get('topic_shift')])}개 — 재작성 효과 집중",
        shift_results,
    )
    print(
        "\n※ 결과를 README에 기록하고, 멀티턴 rerank 개선폭이 미미하면 "
        "app/services/paper_rag.py의 _RERANK_ENABLED를 False로 내려 지연을 절약할 것."
    )


if __name__ == "__main__":
    main()
