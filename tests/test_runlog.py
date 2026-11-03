import csv
import json

import fraudfl.runlog as runlog


def test_archive_never_overwrites_and_writes_journal(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog, "ARCHIVE", tmp_path / "_luu_tru")
    monkeypatch.setattr(runlog, "JOURNAL", tmp_path / "NHAT_KY_CHAY.csv")
    monkeypatch.setattr(runlog, "ROOT", tmp_path)
    out = tmp_path / "ket_qua"
    out.mkdir()
    (out / "per_fold.csv").write_text("a\n1\n", encoding="utf-8")

    d1 = runlog.archive_run(out, note="lần 1", argv=["x.py", "--folds", "0"])
    (out / "per_fold.csv").write_text("a\n2\n", encoding="utf-8")
    d2 = runlog.archive_run(out, note="lần 2", argv=["x.py"])

    assert d1 != d2
    assert (d1 / "per_fold.csv").read_text(encoding="utf-8") == "a\n1\n"     # bản cũ còn nguyên
    assert json.loads((d1 / "meta.json").read_text(encoding="utf-8"))["lenh"] == "x.py --folds 0"
    rows = list(csv.DictReader(open(tmp_path / "NHAT_KY_CHAY.csv", encoding="utf-8")))
    assert [r["ghi_chu"] for r in rows] == ["lần 1", "lần 2"]
