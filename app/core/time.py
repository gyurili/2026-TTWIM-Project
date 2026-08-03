from datetime import datetime
from zoneinfo import ZoneInfo

KST = ZoneInfo("Asia/Seoul")


def epoch_ms_to_kst_naive(epoch_ms: int) -> datetime:
    """epoch ms를 KST 벽시계 시각으로 변환한다. DB 컬럼이 naive라 tzinfo는 제거."""
    return datetime.fromtimestamp(epoch_ms / 1000, tz=KST).replace(tzinfo=None)
