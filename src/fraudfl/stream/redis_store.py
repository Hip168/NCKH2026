"""Lưu trạng thái đặc trưng trên Redis bằng sorted set: điểm = thời gian, thành viên = "tx_id:giá_trị"."""

import redis

from fraudfl.stream.online_features import HISTORY_SECONDS


class RedisStore:
    def __init__(self, url: str):
        self.r = redis.Redis.from_url(url, decode_responses=True)

    def add(self, key, tx_id, ts, value):
        p = self.r.pipeline(transaction=False)
        p.zadd(key, {f"{tx_id}:{value!r}": ts})
        p.zremrangebyscore(key, "-inf", f"({ts - HISTORY_SECONDS}")
        p.execute()

    def window(self, key, after, until):
        rows = self.r.zrangebyscore(key, f"({after}", until)
        return [float(m.split(":", 1)[1]) for m in rows]

    def add_many(self, rows: list[tuple[str, int, int, float]], batch: int = 5000):
        """Nạp lịch sử hàng loạt (key, tx_id, ts, value)."""
        for i in range(0, len(rows), batch):
            p = self.r.pipeline(transaction=False)
            for key, tx_id, ts, value in rows[i:i + batch]:
                p.zadd(key, {f"{tx_id}:{value!r}": ts})
            p.execute()
