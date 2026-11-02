"""Tính đặc trưng từng giao dịch một, cho kết quả GIỐNG `data/features.py` (bản theo lô).

Mô hình học trên đặc trưng theo lô, nên nếu bản real-time lệch thì mô hình chấm sai mà không báo lỗi.
`tests/test_online_features.py` so hai bản trên từng giao dịch.

Trạng thái (lịch sử giao dịch theo khách và terminal) nằm trong `Store`:
- MemoryStore cho test.
- RedisStore cho hệ thống thật, dùng sorted set: `tx_id` là thành viên nên ghi lại cùng giao dịch không bị đếm
  đôi. Nhờ vậy consumer sập và xử lý lại vẫn đúng.

Nhãn gian lận chỉ được đọc ở giao dịch cũ hơn TERMINAL_DELAY_DAYS, giống đặc trưng rủi ro terminal theo lô.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

from fraudfl.config import TERMINAL_DELAY_DAYS
from fraudfl.data.features import FEATURES, WINDOWS

DAY = 86_400
HISTORY_SECONDS = (max(WINDOWS) + TERMINAL_DELAY_DAYS + 1) * DAY


@dataclass(frozen=True)
class Tx:
    tx_id: int
    bank: int
    customer: int
    terminal: int
    amount: float
    ts: int                 # giây kể từ epoch (UTC)
    fraud: int              # nhãn thật; chỉ dùng cho giao dịch đã quá độ trễ nhãn


class Store(Protocol):
    def add(self, key: str, tx_id: int, ts: int, value: float) -> None: ...
    def window(self, key: str, after: int, until: int) -> list[float]: ...


class MemoryStore:
    def __init__(self):
        self.d: dict[str, dict[int, tuple[int, float]]] = {}

    def add(self, key, tx_id, ts, value):
        self.d.setdefault(key, {})[tx_id] = (ts, value)

    def window(self, key, after, until):
        return [v for t, v in self.d.get(key, {}).values() if after < t <= until]


def keys(tx: Tx) -> tuple[str, str]:
    return f"b{tx.bank}:c{tx.customer}", f"b{tx.bank}:t{tx.terminal}"


def remember(store: Store, tx: Tx) -> None:
    """Ghi giao dịch vào lịch sử. Phải gọi TRƯỚC features(), vì cửa sổ theo lô tính cả giao dịch hiện tại."""
    k_c, k_t = keys(tx)
    store.add(k_c, tx.tx_id, tx.ts, tx.amount)
    store.add(k_t, tx.tx_id, tx.ts, tx.fraud)


def features(store: Store, tx: Tx, delay_days: int = TERMINAL_DELAY_DAYS) -> dict[str, float]:
    k_c, k_t = keys(tx)
    moment = datetime.fromtimestamp(tx.ts, tz=timezone.utc)
    out = {"TX_AMOUNT": tx.amount, "TX_DURING_WEEKEND": float(moment.weekday() >= 5),
           "TX_DURING_NIGHT": float(moment.hour <= 6)}
    for w in WINDOWS:
        amounts = store.window(k_c, tx.ts - w * DAY, tx.ts)
        out[f"CUSTOMER_ID_NB_TX_{w}DAY_WINDOW"] = float(len(amounts))
        out[f"CUSTOMER_ID_AVG_AMOUNT_{w}DAY_WINDOW"] = sum(amounts) / len(amounts)
    out["TX_AMOUNT_RATIO_30DAY"] = tx.amount / max(out["CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW"], 0.01)

    known_until = tx.ts - delay_days * DAY
    for w in WINDOWS:
        labels = store.window(k_t, known_until - w * DAY, known_until)
        out[f"TERMINAL_ID_NB_TX_{w}DAY_WINDOW"] = float(len(labels))
        out[f"TERMINAL_ID_RISK_{w}DAY_WINDOW"] = sum(labels) / len(labels) if labels else 0.0
    return out


def as_vector(row: dict[str, float]) -> list[float]:
    return [row[f] for f in FEATURES]
