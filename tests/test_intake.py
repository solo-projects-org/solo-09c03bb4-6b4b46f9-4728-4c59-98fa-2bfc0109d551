"""Tests for the quarantine intake flow.

Two levels are covered:

* ``parsing.parse_hotspots_csv_quarantine`` — which problems isolate a single
  row and which ones reject the whole file;
* the HTTP/service flow — review page, confirmation, batch/run creation,
  bad-row retention, idempotent confirmation and cross-instance concurrency.
"""

import io
import json
import threading

import pytest
from fastapi.testclient import TestClient

from app import db, service
from app.main import create_app
from app.parsing import InputError, parse_hotspots_csv_quarantine

HEADER = "hotspot_no,lon,lat,observed_at\n"


def make_client(tmp_path, start_worker=False):
    app = create_app(db_path=str(tmp_path / "forest.db"), start_worker=start_worker)
    return TestClient(app), app


def geojson(polygons):
    features = []
    for code, (minx, miny, maxx, maxy) in polygons.items():
        features.append(
            {
                "type": "Feature",
                "properties": {"code": code},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [minx, miny],
                            [maxx, miny],
                            [maxx, maxy],
                            [minx, maxy],
                            [minx, miny],
                        ]
                    ],
                },
            }
        )
    return json.dumps({"type": "FeatureCollection", "features": features}).encode()


def csv_bytes(text: str) -> bytes:
    return text.encode()


def drain(app):
    while True:
        claimed = service.claim_next_run(app.state.settings["db_path"])
        if claimed is None:
            return
        run_id, attempt_token = claimed
        service.execute_run(app.state.settings["db_path"], run_id, attempt_token=attempt_token)


# --------------------------------------------------------------------------
# Parser level
# --------------------------------------------------------------------------


def test_quarantine_isolates_each_bad_row_kind():
    data = csv_bytes(
        HEADER
        + "H-1,100.0,25.0,2026-01-01T00:00:00Z\n"          # valid
        + ",100.0,25.0,2026-01-01T00:01:00Z\n"             # empty hotspot number
        + "H-3,181.0,25.0,2026-01-01T00:02:00Z\n"          # longitude out of range
        + "H-4,100.0,91.0,2026-01-01T00:03:00Z\n"          # latitude out of range
        + "H-5,nope,25.0,2026-01-01T00:04:00Z\n"           # non-numeric lon
        + "H-6,100.0,25.0,not-a-time\n"                    # invalid UTC timestamp
        + "H-7,100.0,,2026-01-01T00:06:00Z\n"              # empty latitude
    )
    result = parse_hotspots_csv_quarantine(data, "x.csv")
    assert result.valid_count == 1
    assert result.bad_count == 6
    source_rows = {row.source_row: row for row in result.bad_rows}
    assert sorted(source_rows) == [3, 4, 5, 6, 7, 8]
    assert "热点号为空" in source_rows[3].reason
    assert "经度" in source_rows[4].reason and "超出范围" in source_rows[4].reason
    assert "纬度" in source_rows[5].reason and "超出范围" in source_rows[5].reason
    assert "不是数字" in source_rows[6].reason
    assert "UTC 时刻无效" in source_rows[7].reason
    # Original values are retained for review.
    assert source_rows[3].raw["hotspot_no"] == ""
    assert source_rows[7].raw["observed_at"] == "not-a-time"
    # Valid rows keep normalised timestamps and source line numbers.
    valid = result.hotspots[0]
    assert valid.hotspot_no == "H-1" and valid.source_row == 2 and valid.seq == 0


def test_quarantine_bad_row_can_report_multiple_reasons_at_once():
    data = csv_bytes(
        HEADER
        + "H-0,100.0,25.0,2026-01-01T00:00:00Z\n"  # one valid row keeps the file alive
        + ",200.0,200.0,garbage\n"
    )
    result = parse_hotspots_csv_quarantine(data, "x.csv")
    assert result.valid_count == 1 and result.bad_count == 1
    reason = result.bad_rows[0].reason
    assert "热点号为空" in reason
    assert "经度" in reason and "纬度" in reason
    assert "UTC 时刻无效" in reason


def test_quarantine_whole_file_rejects_missing_required_column():
    with pytest.raises(InputError, match="缺少必需列"):
        parse_hotspots_csv_quarantine(csv_bytes("hotspot_no,lon,lat\nH-1,1,1\n"), "x.csv")


def test_quarantine_whole_file_rejects_column_count_mismatch():
    data = csv_bytes(
        HEADER
        + "H-1,100.0,25.0,2026-01-01T00:00:00Z\n"
        + "H-2,100.0,25.0\n"  # one column short
    )
    with pytest.raises(InputError, match="列数"):
        parse_hotspots_csv_quarantine(data, "x.csv")
    # Extra columns are likewise a structural mismatch.
    data_extra = csv_bytes(HEADER + "H-1,100.0,25.0,2026-01-01T00:00:00Z,extra\n")
    with pytest.raises(InputError, match="列数"):
        parse_hotspots_csv_quarantine(data_extra, "x.csv")


def test_quarantine_whole_file_rejects_when_no_valid_row():
    data = csv_bytes(HEADER + ",1,1,2026-01-01T00:00:00Z\n" + "H-2,1,1,bad\n")
    with pytest.raises(InputError, match="整份文件拒收"):
        parse_hotspots_csv_quarantine(data, "x.csv")


def test_quarantine_skips_blank_lines_and_keeps_source_row_numbers():
    data = csv_bytes(
        HEADER
        + "H-1,100.0,25.0,2026-01-01T00:00:00Z\n"
        + "\n"
        + "   \n"
        + "H-2,100.0,26.0,2026-01-01T00:02:00Z\n"
    )
    result = parse_hotspots_csv_quarantine(data, "x.csv")
    assert result.valid_count == 2 and result.bad_count == 0
    assert [row.source_row for row in result.hotspots] == [2, 5]


def test_quarantine_accepts_chinese_headers_and_extra_columns():
    data = csv_bytes(
        "热点号,经度,纬度,UTC时刻,置信度\n"
        "H-1,100.1,25.1,2026-01-01T00:00:00Z,high\n"
        ",100.1,25.1,2026-01-01T00:01:00Z,low\n"
    )
    result = parse_hotspots_csv_quarantine(data, "x.csv")
    assert result.valid_count == 1 and result.bad_count == 1
    assert result.hotspots[0].raw["置信度"] == "high"
    assert result.bad_rows[0].raw["置信度"] == "low"


def test_strict_parser_path_is_unchanged():
    # The original strict upload still rejects the first bad row outright.
    from app.parsing import parse_hotspots_csv

    with pytest.raises(InputError, match="热点号"):
        parse_hotspots_csv(csv_bytes(HEADER + ",1,1,2026-01-01T00:00:00Z\n"), "x.csv")


# --------------------------------------------------------------------------
# HTTP / service level
# --------------------------------------------------------------------------


def _upload_map(client):
    return client.post(
        "/maps",
        files={"file": ("m.geojson", geojson({"A": (0, 0, 2, 2)}), "application/geo+json")},
        follow_redirects=False,
    )


def _mixed_csv() -> bytes:
    return csv_bytes(
        HEADER
        + "H-1,0.5,0.5,2026-01-01T00:00:00Z\n"
        + "H-2,0.6,0.6,2026-01-01T00:01:00Z\n"
        + "H-3,181,0.5,2026-01-01T00:02:00Z\n"      # bad lon -> isolated
        + ",0.6,0.6,2026-01-01T00:03:00Z\n"         # empty hotspot id -> isolated
        + "H-5,0.6,0.6,nope\n"                      # bad timestamp -> isolated
    )


def test_intake_upload_lists_valid_and_bad_rows(tmp_path):
    client, _ = make_client(tmp_path)
    with client:
        r = client.post(
            "/intakes",
            files={"file": ("mixed.csv", _mixed_csv(), "text/csv")},
            follow_redirects=False,
        )
        assert r.status_code == 303 and r.headers["location"].startswith("/intakes/1")

        page = client.get("/intakes/1")
        assert page.status_code == 200
        text = page.text
        assert "2 行有效" in text or "有效行" in text
        # Row numbers, reasons and original values are all visible.
        assert ">4<" in text and ">5<" in text and ">6<" in text
        assert "超出范围" in text and "热点号为空" in text and "UTC 时刻无效" in text
        assert "nope" in text

        # No batch exists yet: confirmation is an explicit second step.
        with db.connection(str(tmp_path / "forest.db")) as c:
            assert c.execute("SELECT COUNT(*) AS n FROM batches").fetchone()["n"] == 0
            intake = service.get_intake(c, 1)
            assert intake["status"] == "pending"
            assert service.list_intake_bad_rows(c, 1)[0]["source_row"] == 4


def test_intake_file_level_rejection_creates_nothing(tmp_path):
    client, _ = make_client(tmp_path)
    with client:
        r = client.post(
            "/intakes",
            files={"file": ("bad.csv", b"hotspot_no,lon,lat\nH-1,1,1\n", "text/csv")},
            follow_redirects=False,
        )
        assert r.status_code == 303 and "err=" in r.headers["location"]
        with db.connection(str(tmp_path / "forest.db")) as c:
            assert c.execute("SELECT COUNT(*) AS n FROM intakes").fetchone()["n"] == 0


def test_confirm_without_active_map_is_blocked(tmp_path):
    client, _ = make_client(tmp_path)
    with client:
        client.post("/intakes", files={"file": ("mixed.csv", _mixed_csv(), "text/csv")})
        r = client.post("/intakes/1/confirm", follow_redirects=False)
        assert r.status_code == 303 and "err=" in r.headers["location"]
        page = client.get("/intakes/1")
        assert "没有活动林班底图" in page.text or "先" in page.text
        with db.connection(str(tmp_path / "forest.db")) as c:
            assert c.execute("SELECT COUNT(*) AS n FROM batches").fetchone()["n"] == 0
            assert service.get_intake(c, 1)["status"] == "pending"


def test_confirm_creates_batch_with_first_run_and_only_valid_rows(tmp_path):
    client, app = make_client(tmp_path)
    with client:
        _upload_map(client)
        client.post("/intakes", files={"file": ("mixed.csv", _mixed_csv(), "text/csv")})
        r = client.post("/intakes/1/confirm", follow_redirects=False)
        assert r.status_code == 303 and r.headers["location"].startswith("/batches/1")

        drain(app)
        payload = client.get("/api/batches/1").json()
        # Only the two valid hotspots entered attribution; bad rows never did.
        assert payload["published_total"] == 2
        assert payload["published_map_version"] == 1
        assert payload["runs"][0]["status"] == "succeeded"

        # The batch page retains the bad rows.
        page = client.get("/batches/1").text
        assert "随批次留存的坏行" in page
        assert "热点号为空" in page and "nope" in page

        # Exported detail contains exactly the valid source rows.
        csv_text = client.get("/batches/1/download.csv").content.decode("utf-8-sig")
        lines = csv_text.splitlines()
        assert len(lines) == 3
        assert "H-1" in lines[1] and "H-2" in lines[2]
        assert "H-3" not in csv_text and "nope" not in csv_text


def test_bad_rows_remain_excluded_after_recompute(tmp_path):
    client, app = make_client(tmp_path)
    with client:
        _upload_map(client)
        client.post("/intakes", files={"file": ("mixed.csv", _mixed_csv(), "text/csv")})
        client.post("/intakes/1/confirm")
        drain(app)

        # Swap the map and recompute: the batch must still cover only valid rows.
        client.post(
            "/maps",
            files={"file": ("m2.geojson", geojson({"B": (10, 10, 12, 12)}), "application/geo+json")},
        )
        client.post("/batches/1/recompute")
        drain(app)
        payload = client.get("/api/batches/1").json()
        assert payload["published_map_version"] == 2
        assert payload["published_total"] == 2
        with db.connection(str(tmp_path / "forest.db")) as c:
            assert c.execute("SELECT COUNT(*) AS n FROM hotspots WHERE batch_id=1").fetchone()["n"] == 2
            assert len(service.list_batch_bad_rows(c, 1)) == 3


def test_duplicate_confirm_is_idempotent(tmp_path):
    client, app = make_client(tmp_path)
    with client:
        _upload_map(client)
        client.post("/intakes", files={"file": ("mixed.csv", _mixed_csv(), "text/csv")})
        first = client.post("/intakes/1/confirm", follow_redirects=False)
        second = client.post("/intakes/1/confirm", follow_redirects=False)
        assert first.status_code == second.status_code == 303
        with db.connection(str(tmp_path / "forest.db")) as c:
            assert c.execute("SELECT COUNT(*) AS n FROM batches").fetchone()["n"] == 1
            assert c.execute("SELECT COUNT(*) AS n FROM runs").fetchone()["n"] == 1
            intake = service.get_intake(c, 1)
            assert intake["status"] == "confirmed" and intake["batch_id"] == 1
        drain(app)


def test_concurrent_confirm_from_two_threads_makes_one_batch_and_run(tmp_path):
    """Two web instances confirming the same pending intake at once."""

    db_path = str(tmp_path / "forest.db")
    client, app = make_client(tmp_path)
    with client:
        _upload_map(client)
        client.post("/intakes", files={"file": ("mixed.csv", _mixed_csv(), "text/csv")})

    outcomes: list[tuple[int, int | None]] = []
    errors: list[BaseException] = []

    def confirm():
        try:
            outcomes.append(service.confirm_intake(db_path, 1))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=confirm) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors
    assert outcomes == [(1, 1), (1, 1)]
    with db.connection(db_path) as c:
        assert c.execute("SELECT COUNT(*) AS n FROM batches").fetchone()["n"] == 1
        assert c.execute("SELECT COUNT(*) AS n FROM runs").fetchone()["n"] == 1
        assert c.execute("SELECT COUNT(*) AS n FROM hotspots").fetchone()["n"] == 2
        assert c.execute("SELECT COUNT(*) AS n FROM intake_bad_rows").fetchone()["n"] == 3
        assert c.execute(
            "SELECT COUNT(*) AS n FROM intakes WHERE status='confirmed'"
        ).fetchone()["n"] == 1


def test_interrupted_confirmation_leaves_no_orphan_batch(tmp_path):
    """A failure inside the confirmation transaction rolls back everything."""

    db_path = str(tmp_path / "forest.db")
    client, _ = make_client(tmp_path)
    with client:
        _upload_map(client)
        client.post("/intakes", files={"file": ("mixed.csv", _mixed_csv(), "text/csv")})

    import app.service as service_module

    original = service_module._insert_run

    def boom(conn, batch_id, map_version_id):  # simulate a crash mid-confirmation
        raise RuntimeError("simulated interruption")

    service_module._insert_run = boom
    try:
        with pytest.raises(RuntimeError, match="simulated interruption"):
            service.confirm_intake(db_path, 1)
    finally:
        service_module._insert_run = original

    with db.connection(db_path) as c:
        assert c.execute("SELECT COUNT(*) AS n FROM batches").fetchone()["n"] == 0
        assert c.execute("SELECT COUNT(*) AS n FROM runs").fetchone()["n"] == 0
        assert c.execute("SELECT COUNT(*) AS n FROM hotspots").fetchone()["n"] == 0
        intake = service.get_intake(c, 1)
        assert intake["status"] == "pending" and intake["batch_id"] is None

    # Retrying after the interruption succeeds.
    batch_id, run_id = service.confirm_intake(db_path, 1)
    assert (batch_id, run_id) == (1, 1)
    with db.connection(db_path) as c:
        assert service.get_intake(c, 1)["status"] == "confirmed"


def test_strict_upload_route_still_works_alongside_intake(tmp_path):
    client, app = make_client(tmp_path)
    with client:
        _upload_map(client)
        good = (
            HEADER
            + "H-1,0.5,0.5,2026-01-01T00:00:00Z\n"
            + "H-2,0.6,0.6,2026-01-01T00:01:00Z\n"
        ).encode()
        r = client.post(
            "/batches", files={"file": ("good.csv", good, "text/csv")}, follow_redirects=False
        )
        assert r.status_code == 303 and r.headers["location"].startswith("/batches/1")
        drain(app)
        assert client.get("/api/batches/1").json()["published_total"] == 2

        # Strict route still rejects a bad row instead of quarantining.
        r = client.post(
            "/batches",
            files={"file": ("bad.csv", _mixed_csv(), "text/csv")},
            follow_redirects=False,
        )
        assert r.status_code == 303 and "err=" in r.headers["location"]


def test_intake_review_page_404_and_empty_file_rejection(tmp_path):
    client, _ = make_client(tmp_path)
    with client:
        assert client.get("/intakes/999").status_code == 404
        r = client.post(
            "/intakes", files={"file": ("empty.csv", b"", "text/csv")}, follow_redirects=False
        )
        assert r.status_code == 303 and "err=" in r.headers["location"]
