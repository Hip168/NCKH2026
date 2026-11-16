import numpy as np
import pandas as pd
import pytest

from fraudfl.data.quoc_gia import nuoc_tu_ten
from fraudfl.khung import N_FOLD, tang_phan_tang, tao_chia, vai_tro


def _du_lieu(n=20_000, seed=0):
    rng = np.random.default_rng(seed)
    nhom = rng.choice(["a", "b", "c", "d"], size=n, p=[0.5, 0.3, 0.15, 0.05])
    y = (rng.random(n) < np.where(nhom == "a", 0.01, 0.002)).astype(int)
    tx = rng.permutation(n) + 1000                          # TX_ID không theo thứ tự
    return tx, y, nhom


def test_moi_dong_dung_mot_fold_va_tat_dinh():
    tx, y, nhom = _du_lieu()
    a, b = tao_chia(tx, y, nhom, seeds=(42, 7)), tao_chia(tx, y, nhom, seeds=(42, 7))
    pd.testing.assert_frame_equal(a, b)                      # chạy lại ra y hệt
    assert a["TX_ID"].is_monotonic_increasing and len(a) == len(tx)
    for c in ("fold_s42", "fold_s7"):
        assert set(a[c].unique()) == set(range(N_FOLD))
    assert (a["fold_s42"] != a["fold_s7"]).mean() > 0.5      # seed khác → chia khác


def test_vai_tro_khong_trung_va_phu_het():
    tx, y, nhom = _du_lieu()
    fold = tao_chia(tx, y, nhom, seeds=(42,))["fold_s42"].to_numpy()
    test_seen = np.zeros(len(fold), int)
    for k in range(N_FOLD):
        m = vai_tro(fold, k)
        tong = m["fit"].astype(int) + m["val"] + m["test"]
        assert (tong == 1).all()                             # mỗi dòng đúng 1 vai trò
        assert 0.18 < m["val"].mean() < 0.22 and 0.18 < m["test"].mean() < 0.22
        test_seen += m["test"]
    assert (test_seen == 1).all()                            # mỗi dòng làm test đúng 1 lần


def test_phan_tang_chia_deu_so_ca():
    tx, y, nhom = _du_lieu()
    d = tao_chia(tx, y, nhom, seeds=(42,))
    yy = pd.Series(y, index=tx).reindex(d["TX_ID"]).to_numpy()
    gg = pd.Series(nhom, index=tx).reindex(d["TX_ID"]).to_numpy()
    ca_a = np.bincount(d["fold_s42"][gg == "a"], weights=yy[gg == "a"], minlength=N_FOLD)
    assert ca_a.max() - ca_a.min() <= 1                       # nhóm có >= 10 ca: chia đều tới từng ca
    ca = np.bincount(d["fold_s42"], weights=yy, minlength=N_FOLD)
    assert ca.max() - ca.min() <= 2


def test_tang_nhom_it_ca_gop_theo_nhan():
    s = tang_phan_tang(np.array([1, 0, 0, 1]), np.array(["x", "x", "y", "y"]))
    assert (s < 0).all()                                     # không nhóm nào đủ 10 ca → chỉ theo nhãn


def test_trung_tx_id_bao_loi():
    with pytest.raises(ValueError):
        tao_chia(np.array([1, 1, 2]), np.array([0, 1, 0]), np.array(["a", "a", "a"]), seeds=(42,))


def test_nuoc_tu_ten():
    ten = pd.Series(["Germany Bank #123", "Saudi Arabia Bank #7", "National Bank of Cleveland", "Crytpo Bank #9"])
    assert list(nuoc_tu_ten(ten)) == ["Germany", "Saudi Arabia", "USA", "Crypto"]


def test_client_theo_nuoc_gop_nuoc_nho_va_ma_on_dinh():
    from fraudfl.data.quoc_gia import TEN_GOP, client_theo_nuoc

    nuoc = pd.Series(["A"] * 50 + ["B"] * 30 + ["C"] * 5 + ["D"] * 3)
    b = client_theo_nuoc(nuoc, gop_nho=10)
    m = b.set_index("NUOC")
    assert m.loc["A", "MA_CLIENT"] == 0 and m.loc["B", "MA_CLIENT"] == 1     # nhiều giao dịch nhất mã 0
    assert m.loc["C", "CLIENT"] == m.loc["D", "CLIENT"] == TEN_GOP           # nước nhỏ gộp chung
    assert m.loc["C", "MA_CLIENT"] == b["MA_CLIENT"].max()                    # client gộp luôn ở cuối
    pd.testing.assert_frame_equal(b, client_theo_nuoc(nuoc.sample(frac=1, random_state=1), gop_nho=10))
    with pytest.raises(ValueError):
        client_theo_nuoc(pd.Series(["A", None]))


def test_script12_chia_b0_dung_vai_tro_cua_khung():
    import importlib.util
    import sys
    from argparse import Namespace
    from pathlib import Path

    from fraudfl.khung import FILE_CHIA, gan_fold

    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location("s12", scripts / "12_ibm_cach1_lo_nho.py")
    s12 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s12)

    tx = pd.read_parquet(FILE_CHIA, columns=["TX_ID"])["TX_ID"].to_numpy()[::997]   # ~5 nghìn giao dịch thật
    tx = np.random.default_rng(0).permutation(tx)                                     # thứ tự dòng bất kỳ
    y = np.zeros(len(tx), dtype=np.int64)
    cid = np.zeros(len(tx), dtype=np.int64)
    fold = gan_fold(tx, 7)
    ra = list(s12.chia_fold(Namespace(chia="b0", seed=7), tx, y, cid, 1))
    assert len(ra) == N_FOLD
    for k, (fit, val, te) in enumerate(ra):
        vt = vai_tro(fold, k)
        assert np.array_equal(fit, np.flatnonzero(vt["fit"]))
        assert np.array_equal(val, np.flatnonzero(vt["val"]))
        assert np.array_equal(te, np.flatnonzero(vt["test"]))
