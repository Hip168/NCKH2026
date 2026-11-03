from fraudfl.data.ibm_aml import load_raw

SAMPLE = """Timestamp,From Bank,Account,To Bank,Account,Amount Received,Receiving Currency,Amount Paid,Payment Currency,Payment Format,Is Laundering
2022/09/01 00:20,10,8000EBD30,10,8000EBD31,100.5,US Dollar,100.5,US Dollar,Reinvestment,0
2022/09/02 00:20,3208,8000F4580,1,8000F5340,0.01,US Dollar,0.01,US Dollar,Cheque,1
"""


def test_load_raw_keeps_both_account_columns(tmp_path):
    p = tmp_path / "x_Trans.csv"
    p.write_text(SAMPLE, encoding="utf-8")
    df = load_raw(p)
    assert df["From Account"].tolist() == ["8000EBD30", "8000F4580"]
    assert df["To Account"].tolist() == ["8000EBD31", "8000F5340"]
    assert df["TX_TIME_DAYS"].tolist() == [0, 1]


def test_load_raw_normalizes_leading_zeros_in_bank_codes(tmp_path):
    # "010" và "10" là cùng ngân hàng (đối chiếu accounts.csv) → cùng mã sau chuẩn hóa; mã "0" giữ nguyên
    p = tmp_path / "x_Trans.csv"
    p.write_text(SAMPLE.replace("00:20,10,", "00:20,010,").replace(",3208,", ",0,"), encoding="utf-8")
    df = load_raw(p)
    assert df["From Bank"].tolist() == ["10", "0"]
    assert df["To Bank"].tolist() == ["10", "1"]
