"""Cross-process concurrency tests: real Uvicorn servers, one database.

These exercise the guarantee that matters in deployment: two web instances
uploading/recomputing at the same time must never publish a detail file that
mixes results from two different compartment versions.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class Server:
    def __init__(self, db_path: Path, port: int, cwd: Path):
        self.port = port
        self.base = f"http://127.0.0.1:{port}"
        env = dict(os.environ)
        env.update(
            FOREST_DB_PATH=str(db_path),
            FOREST_RUN_WORKER="1",
            FOREST_STALE_RUN_SECONDS="2",
            PYTHONPATH=str(ROOT),
        )
        self.proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "app.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--log-level",
                "warning",
            ],
            cwd=cwd,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def wait_ready(self, timeout: float = 25.0) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                if httpx.get(self.base + "/healthz", timeout=1).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.1)
        raise RuntimeError("server did not start")

    def stop(self) -> None:
        self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover
            self.proc.kill()


def upload_map(client: httpx.Client, name: str, payload: bytes) -> None:
    response = client.post(
        f"{client.base_url}/maps",
        files={"file": (name, payload, "application/geo+json")},
        follow_redirects=False,
    )
    assert response.status_code == 303, response.text


def upload_hotspots(client: httpx.Client, payload: bytes) -> None:
    response = client.post(
        f"{client.base_url}/batches",
        files={"file": ("hotspots.csv", payload, "text/csv")},
        follow_redirects=False,
    )
    assert response.status_code == 303, response.text


def wait_published(client: httpx.Client, batch_id: int, timeout: float = 25.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        payload = client.get(f"{client.base_url}/api/batches/{batch_id}").json()
        if payload["published_total"] is not None and not payload["pending"]:
            return payload
        time.sleep(0.05)
    raise AssertionError("batch never published")


@pytest.mark.slow
def test_two_servers_publish_single_consistent_version(tmp_path):
    db_path = tmp_path / "forest.db"
    port_a, port_b = free_port(), free_port()
    server_a = Server(db_path, port_a, tmp_path)
    server_b = Server(db_path, port_b, tmp_path)
    server_a.wait_ready()
    server_b.wait_ready()
    try:
        with httpx.Client(base_url=server_a.base, timeout=30) as client_a, httpx.Client(
            base_url=server_b.base, timeout=30
        ) as client_b:
            upload_map(client_a, "v1.geojson", (SAMPLES / "compartments_v1.geojson").read_bytes())
            upload_hotspots(client_b, (SAMPLES / "hotspots.csv").read_bytes())
            first = wait_published(client_a, 1)
            assert first["published_map_version"] == 1

            # Instance A swaps the map while instance B triggers a recompute.
            upload_map(client_a, "v2.geojson", (SAMPLES / "compartments_v2.geojson").read_bytes())
            response = client_b.post(
                f"{client_b.base_url}/batches/1/recompute", follow_redirects=False
            )
            assert response.status_code == 303

            final = wait_published(client_b, 1)
            assert final["published_map_version"] == 2
            assert final["published_total"] == 15

            # The published CSV must be internally consistent: it is produced
            # exclusively from the run pinned to the active v2 map.
            csv_bytes = client_a.get(f"{client_a.base_url}/batches/1/download.csv").content
            text = csv_bytes.decode("utf-8-sig")
            rows = text.splitlines()
            assert len(rows) == 16
            assert all("v2" in line for line in rows[1:])
            assert "v1" not in text
            assert "A-01;B-02" not in text  # v1-only shared edge between A and B

            # Every hotspot row present exactly once.
            assert text.count("H-00") == 15

            # No computation against the retired v1 map may remain pending, and
            # no v1 run may be the published one.
            payload = client_a.get(f"{client_a.base_url}/api/batches/1").json()
            active_version = client_a.get(f"{client_a.base_url}/healthz").json()[
                "active_map_version"
            ]
            assert payload["published_map_version"] == active_version == 2
            for run in payload["runs"]:
                if run["map_version_id"] == 1:
                    assert run["status"] in {"succeeded", "superseded"}
            published_runs = [
                run for run in payload["runs"] if run["id"] == payload["published_run_id"]
            ]
            assert published_runs[0]["map_version_id"] == active_version
    finally:
        server_a.stop()
        server_b.stop()


@pytest.mark.slow
def test_concurrent_map_swaps_leave_exactly_one_active(tmp_path):
    db_path = tmp_path / "forest.db"
    port_a, port_b = free_port(), free_port()
    server_a = Server(db_path, port_a, tmp_path)
    server_b = Server(db_path, port_b, tmp_path)
    server_a.wait_ready()
    server_b.wait_ready()
    try:
        with httpx.Client(base_url=server_a.base, timeout=30) as client_a, httpx.Client(
            base_url=server_b.base, timeout=30
        ) as client_b:
            upload_hotspots(client_a, (SAMPLES / "hotspots.csv").read_bytes())
            v1 = (SAMPLES / "compartments_v1.geojson").read_bytes()
            v2 = (SAMPLES / "compartments_v2.geojson").read_bytes()

            import threading

            errors: list[BaseException] = []

            def swap(client, payload, name):
                try:
                    upload_map(client, name, payload)
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)

            threads = [
                threading.Thread(target=swap, args=(client_a, v1, "a.geojson")),
                threading.Thread(target=swap, args=(client_b, v2, "b.geojson")),
                threading.Thread(target=swap, args=(client_a, v2, "c.geojson")),
                threading.Thread(target=swap, args=(client_b, v1, "d.geojson")),
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            assert not errors

            health = client_a.get(f"{client_a.base_url}/healthz").json()
            assert health["active_map_version"] is not None
            from app import db, service

            with db.connection(str(db_path)) as conn:
                active_count = conn.execute(
                    "SELECT COUNT(*) AS n FROM map_versions WHERE is_active = 1"
                ).fetchone()["n"]
                versions = conn.execute("SELECT COUNT(*) AS n FROM map_versions").fetchone()["n"]
            assert active_count == 1
            assert versions == 4

            # The active version is the one the health endpoint reports, and a
            # recompute publishes against it with no mixed output.
            client_b.post(f"{client_b.base_url}/batches/1/recompute", follow_redirects=False)
            final = wait_published(client_b, 1)
            assert final["published_map_version"] == health["active_map_version"]
    finally:
        server_a.stop()
        server_b.stop()


def _mixed_csv_bytes() -> bytes:
    return (
        "hotspot_no,lon,lat,observed_at\n"
        "H-1,100.05,25.05,2026-03-01T02:10:00Z\n"
        "H-2,100.08,25.05,2026-03-01T02:12:30Z\n"
        "H-3,181.0,25.05,2026-03-01T02:14:00Z\n"          # bad longitude, isolated
        ",100.08,25.08,2026-03-01T02:15:00Z\n"            # empty hotspot id, isolated
        "H-5,100.10,25.10,not-a-time\n"                   # bad timestamp, isolated
    ).encode()


@pytest.mark.slow
def test_two_servers_confirm_same_intake_once(tmp_path):
    """Concurrent confirmation on two instances yields one batch and one run."""

    db_path = tmp_path / "forest.db"
    port_a, port_b = free_port(), free_port()
    server_a = Server(db_path, port_a, tmp_path)
    server_b = Server(db_path, port_b, tmp_path)
    server_a.wait_ready()
    server_b.wait_ready()
    try:
        with httpx.Client(base_url=server_a.base, timeout=30) as client_a, httpx.Client(
            base_url=server_b.base, timeout=30
        ) as client_b:
            upload_map(client_a, "v1.geojson", (SAMPLES / "compartments_v1.geojson").read_bytes())

            response = client_a.post(
                f"{client_a.base_url}/intakes",
                files={"file": ("mixed.csv", _mixed_csv_bytes(), "text/csv")},
                follow_redirects=False,
            )
            assert response.status_code == 303
            assert response.headers["location"].startswith("/intakes/1")

            review = client_a.get(f"{client_a.base_url}/intakes/1")
            assert review.status_code == 200
            assert "2" in review.text and "热点号为空" in review.text

            import threading

            redirects: list[httpx.Response] = []
            errors: list[BaseException] = []

            def confirm(client):
                try:
                    redirects.append(
                        client.post(
                            f"{client.base_url}/intakes/1/confirm", follow_redirects=False
                        )
                    )
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)

            threads = [
                threading.Thread(target=confirm, args=(client_a,)),
                threading.Thread(target=confirm, args=(client_b,)),
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            assert not errors
            assert all(r.status_code == 303 for r in redirects)
            assert all(r.headers["location"].startswith("/batches/1") for r in redirects)

            final = wait_published(client_a, 1)
            # Only the two valid rows entered attribution; the three bad rows
            # stay with the batch but out of every run.
            assert final["published_total"] == 2
            assert len(final["runs"]) == 1
            assert final["runs"][0]["status"] == "succeeded"

            from app import db as db_module

            with db_module.connection(str(db_path)) as conn:
                assert conn.execute("SELECT COUNT(*) AS n FROM batches").fetchone()["n"] == 1
                assert conn.execute("SELECT COUNT(*) AS n FROM runs").fetchone()["n"] == 1
                assert conn.execute("SELECT COUNT(*) AS n FROM hotspots").fetchone()["n"] == 2
                assert (
                    conn.execute("SELECT COUNT(*) AS n FROM intake_bad_rows").fetchone()["n"] == 3
                )
                intake = conn.execute("SELECT * FROM intakes WHERE id=1").fetchone()
                assert intake["status"] == "confirmed" and intake["batch_id"] == 1

            # Bad rows are still visible on the batch page after confirmation.
            batch_page = client_b.get(f"{client_b.base_url}/batches/1").text
            assert "随批次留存的坏行" in batch_page and "not-a-time" in batch_page
    finally:
        server_a.stop()
        server_b.stop()
