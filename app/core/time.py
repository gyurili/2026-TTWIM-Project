from datetime import datetime
from zoneinfo import ZoneInfo

KST = ZoneInfo("Asia/Seoul")


def epoch_ms_to_kst_naive(epoch_ms: int) -> datetime:
    """epoch ms를 KST 시각으로 변환한다. DB 컬럼이 naive라 tzinfo는 제거."""
    return datetime.fromtimestamp(epoch_ms / 1000, tz=KST).replace(tzinfo=None)


def kst_naive_to_epoch_ms(dt: datetime) -> int:
    """KST 벽시계 시각(naive)을 epoch ms로 변환한다. epoch_ms_to_kst_naive의 역변환.
    서버 프로세스의 로컬 타임존에 의존하지 않도록 tzinfo를 명시적으로 KST로 지정한다."""
    return int(dt.replace(tzinfo=KST).timestamp() * 1000)
def now_kst_naive() -> datetime:
    """현재 시각을 KST 시각으로 반환한다. DB 컬럼이 naive라 tzinfo는 제거."""
    return datetime.now(KST).replace(tzinfo=None)
