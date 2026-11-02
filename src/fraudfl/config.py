"""Cấu hình chung của dự án. Mọi script đọc đường dẫn và tham số từ đây để cả nhóm dùng chung một chuẩn."""

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_SPLITS = ROOT / "data" / "splits"
RESULTS = ROOT / "results"

# IBM AML LI-Small bản đã định dạng, có sẵn trên máy dev (Test/TH1). Máy khác: đặt file vào data/raw/ibm_aml/.
IBM_LI_SMALL_FORMATTED = next(
    (p for p in [DATA_RAW / "ibm_aml" / "LI_Small" / "formatted_transactions.csv",
                 ROOT.parent / "Test" / "TH1" / "data-raw" / "LI_Small" / "formatted_transactions.csv"]
     if p.exists()),
    DATA_RAW / "ibm_aml" / "LI_Small" / "formatted_transactions.csv",
)

SEED = 42

# --- Đánh giá (đã chốt trong kế hoạch, KHÔNG đổi sau khi xem kết quả) ---
ALERT_FRACS = (0.01, 0.02)   # ngân sách cảnh báo: tỷ lệ giao dịch mỗi ngày chuyên viên kiểm tra được (chính 1%, độ nhạy 2%)
BURN_IN_DAYS = 30            # bỏ 30 ngày đầu khỏi train/test: kịch bản gian lận và cửa sổ 30 ngày chưa ổn định
N_OUTER_FOLDS = 5
N_INNER_FOLDS = 3
TERMINAL_DELAY_DAYS = 7      # nhãn gian lận chỉ được biết sau 7 ngày (dùng cho đặc trưng rủi ro terminal)
IBM_MAX_DAY = 9              # D18: IBM HI-Small chỉ dùng ngày 0–9 (ngày 10–17 ~60% rửa tiền — lỗi bộ sinh dữ liệu)


@dataclass
class BankProfile:
    """Hồ sơ một ngân hàng (client). Các tham số khác nhau giữa ngân hàng tạo ra dữ liệu non-IID."""
    name: str
    center: tuple[float, float]          # vùng địa lý tập trung khách hàng trên lưới 100x100
    spread: float                        # độ phân tán khách hàng quanh tâm
    amount_range: tuple[float, float]    # khoảng giá trị trung bình mỗi giao dịch của khách
    rate_range: tuple[float, float]      # khoảng số giao dịch/ngày trung bình của khách
    target_n_tx: int = 100_000           # số giao dịch mong muốn của ngân hàng
    s1_threshold: float = 220.0          # kịch bản 1: giao dịch vượt ngưỡng là gian lận
    s3_customers_per_day: int = 2        # kịch bản 3: số khách bị lộ thông tin thẻ mỗi ngày


@dataclass
class SimulationConfig:
    n_days: int = 120
    start_date: str = "2025-01-01"
    n_terminals: int = 10_000            # terminal (điểm bán) dùng CHUNG giữa các ngân hàng
    terminal_radius: float = 5.0
    s2_terminals_per_day: int = 2        # kịch bản 2: terminal bị xâm nhập mỗi ngày (ảnh hưởng mọi ngân hàng)
    s2_duration_days: int = 28
    s3_duration_days: int = 14
    seed: int = SEED
    banks: list[BankProfile] = field(default_factory=lambda: [
        BankProfile("B0_urban_retail", (30, 30), 18, (5, 60), (0.3, 1.5), s1_threshold=130, s3_customers_per_day=3),
        BankProfile("B1_premium", (70, 70), 15, (50, 200), (0.1, 0.8), s1_threshold=420, s3_customers_per_day=2),
        BankProfile("B2_rural", (25, 75), 15, (5, 40), (0.05, 0.5), s1_threshold=90, s3_customers_per_day=6),
        BankProfile("B3_sme", (75, 25), 15, (20, 120), (0.3, 1.2), s1_threshold=260, s3_customers_per_day=3),
        BankProfile("B4_digital", (50, 50), 30, (5, 80), (0.5, 2.0), s1_threshold=170, s3_customers_per_day=2),
    ])
