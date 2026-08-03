from datetime import datetime
from zoneinfo import ZoneInfo

KST = ZoneInfo("Asia/Seoul")


def epoch_ms_to_kst_naive(epoch_ms: int) -> datetime:
    """epoch ms를 KST 시각으로 변환한다. DB 컬럼이 naive라 tzinfo는 제거."""
    return datetime.fromtimestamp(epoch_ms / 1000, tz=KST).replace(tzinfo=None)


def now_kst_naive() -> datetime:
    """현재 시각을 KST 시각으로 반환한다. DB 컬럼이 naive라 tzinfo는 제거."""
    return datetime.now(KST).replace(tzinfo=None)
