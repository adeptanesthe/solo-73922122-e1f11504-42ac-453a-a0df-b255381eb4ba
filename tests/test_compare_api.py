"""HTTP-level tests for the two-segment comparison endpoints."""

import json

import pytest
from fastapi.testclient import TestClient

from app.main import app

VOCAB = ["<blank>", "a", "b", "c"]


def row(blank=0.0, a=0.0, b=0.0, c=0.0):
    return [blank, a, b, c]


BLANK = row(blank=0.94, a=0.02, b=0.02, c=0.02)
A_HI = row(blank=0.05, a=0.90, b=0.03, c=0.02)
B_HI = row(blank=0.05, a=0.04, b=0.88, c=0.03)
C_HI = row(blank=0.08, a=0.04, b=0.03, c=0.85)


def session_frames():
    # 16 frames; the first 12 contain a,b,c events and lots of blanks.
    return [
        BLANK,
        A_HI, A_HI,
        BLANK,
        B_HI, B_HI,
        BLANK,
        C_HI,
        BLANK, BLANK, BLANK, BLANK,
        BLANK, BLANK, BLANK, BLANK,
    ]


@pytest.fixture()
def client():
    return TestClient(app)


@pytest.fixture()
def sid(client):
    response = client.post(
        "/api/sessions",
        json={"vocab": VOCAB, "beam_width": 4, "frames": session_frames()},
    )
    assert response.status_code == 200, response.text
    session_id = response.json()["session_id"]
    advanced = client.post(
        "/api/advance", json={"session_id": session_id, "to_frame": 16}
    )
    assert advanced.status_code == 200
    return session_id


def good_b():
    return [
        BLANK, BLANK,
        A_HI, A_HI, A_HI,
        BLANK,
        B_HI,
        BLANK,
        C_HI, C_HI,
        BLANK, BLANK, BLANK, BLANK,
    ]


def test_compare_duplicate_pair_returns_ok(client, sid):
    response = client.post(
        "/api/compare",
        json={
            "session_id": sid,
            "start_frame": 0,
            "end_frame": 12,
            "frames_b": good_b(),
            "request_id": "req-1",
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["request_id"] == "req-1"
    assert payload["revision"] >= 0
    assert payload["segment_a"] == {
        "start_frame": 0,
        "end_frame": 12,
        "length": 12,
        "next_frame": 16,
    }
    assert [e["token"] for e in payload["events_a"]] == ["a", "b", "c"]
    assert payload["alignment"]
    assert isinstance(payload["similarity"], float)


def test_compare_accepts_frames_and_probs_wrappers(client, sid):
    for wrapper in (lambda f: {"frames": f}, lambda f: {"probs": f}, lambda f: f):
        response = client.post(
            "/api/compare",
            json={
                "session_id": sid,
                "start_frame": 0,
                "end_frame": 12,
                "frames_b": wrapper(good_b()),
            },
        )
        assert response.status_code == 200, response.text


def test_compare_rejects_unadvanced_frames(client, sid):
    response = client.post(
        "/api/compare",
        json={
            "session_id": sid,
            "start_frame": 0,
            "end_frame": 17,  # beyond the advanced boundary (16 frames)
            "frames_b": good_b(),
        },
    )
    assert response.status_code == 409
    assert "已经推进" in response.json()["detail"]


def test_compare_rejects_bool_and_float_boundaries(client, sid):
    payload = {
        "session_id": sid,
        "start_frame": True,
        "end_frame": 12,
        "frames_b": good_b(),
    }
    assert client.post("/api/compare", json=payload).status_code == 422
    payload["start_frame"] = 0.5
    assert client.post("/api/compare", json=payload).status_code == 422


def test_compare_rejects_empty_or_wrapped_wrong_frames_b_without_500(client, sid):
    for bad in ([], {"foo": []}, {"frames": None}):
        response = client.post(
            "/api/compare",
            json={
                "session_id": sid,
                "start_frame": 0,
                "end_frame": 12,
                "frames_b": bad,
            },
        )
        assert response.status_code in (400, 422), (bad, response.status_code)


def test_compare_rejects_out_of_bounds_length(client, sid):
    too_short = [BLANK] * 7
    response = client.post(
        "/api/compare",
        json={
            "session_id": sid,
            "start_frame": 0,
            "end_frame": 7,
            "frames_b": too_short,
        },
    )
    assert response.status_code in (409, 400)
    response = client.post(
        "/api/compare",
        json={
            "session_id": sid,
            "start_frame": 0,
            "end_frame": 12,
            "frames_b": [BLANK] * 65,
        },
    )
    assert response.status_code == 400


def test_compare_all_zero_b_frames_returns_400_and_keeps_session(client, sid):
    before = client.get(f"/api/sessions/{sid}").json()
    bad_b = [*[BLANK] * 7, [0.0, 0.0, 0.0, 0.0]]
    response = client.post(
        "/api/compare",
        json={
            "session_id": sid,
            "start_frame": 0,
            "end_frame": 12,
            "frames_b": bad_b,
        },
    )
    assert response.status_code == 400
    assert "全零帧" in response.json()["detail"]
    # The failed comparison must not touch the decoding session.
    after = client.get(f"/api/sessions/{sid}").json()
    assert after == before


def test_compare_vocab_width_mismatch_is_400(client, sid):
    response = client.post(
        "/api/compare",
        json={
            "session_id": sid,
            "start_frame": 0,
            "end_frame": 12,
            "frames_b": [[0.5, 0.5, 0.0] for _ in range(10)],
        },
    )
    assert response.status_code == 400


def test_compare_conflict_pair_reports_conflicts(client, sid):
    conflicting_b = [
        BLANK,
        A_HI, A_HI, A_HI,
        BLANK,
        C_HI, C_HI,
        BLANK,
        BLANK, BLANK, BLANK, BLANK, BLANK, BLANK,
    ]
    response = client.post(
        "/api/compare",
        json={
            "session_id": sid,
            "start_frame": 0,
            "end_frame": 12,
            "frames_b": conflicting_b,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "conflict"
    assert payload["conflicts"]


def test_compare_blank_only_segment_is_insufficient(client, sid):
    # Frames 8..16 of the session are all blanks (the stream has 16 frames).
    response = client.post(
        "/api/compare",
        json={
            "session_id": sid,
            "start_frame": 8,
            "end_frame": 16,
            "frames_b": [BLANK] * 10,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "insufficient"
    assert payload["similarity"] is None


def test_compare_upload_endpoint(client, sid):
    files = {
        "frames_b_file": (
            "b.json",
            json.dumps({"frames": good_b()}).encode(),
            "application/json",
        )
    }
    data = {
        "session_id": sid,
        "start_frame": "0",
        "end_frame": "12",
        "request_id": "up-1",
    }
    response = client.post("/api/compare/upload", files=files, data=data)
    assert response.status_code == 200, response.text
    assert response.json()["request_id"] == "up-1"


def test_compare_upload_bad_json_is_400(client, sid):
    files = {"frames_b_file": ("b.json", b"{not json", "application/json")}
    data = {"session_id": sid, "start_frame": "0", "end_frame": "12"}
    response = client.post("/api/compare/upload", files=files, data=data)
    assert response.status_code == 400
    assert "JSON" in response.json()["detail"]


def test_compare_unknown_session_is_404(client):
    response = client.post(
        "/api/compare",
        json={
            "session_id": "nope",
            "start_frame": 0,
            "end_frame": 8,
            "frames_b": good_b(),
        },
    )
    assert response.status_code == 404


def test_compare_samples_are_served(client):
    listing = client.get("/api/compare-samples")
    assert listing.status_code == 200
    names = [item["name"] for item in listing.json()["samples"]]
    assert "pair_duplicate.json" in names
    assert "pair_conflict.json" in names

    sample = client.get("/api/compare-samples/pair_duplicate.json").json()
    assert sample["vocab"][0] == "<blank>"
    assert 8 <= len(sample["frames_a"]) <= 64
    assert 8 <= len(sample["frames_b"]) <= 64

    assert client.get("/api/compare-samples/..%2Fmain.py").status_code == 404


def test_bundled_pair_samples_match_their_intended_verdict(client):
    for name, expected in (
        ("pair_duplicate.json", "ok"),
        ("pair_conflict.json", "conflict"),
    ):
        sample = client.get(f"/api/compare-samples/{name}").json()
        created = client.post(
            "/api/sessions",
            json={
                "vocab": sample["vocab"],
                "beam_width": 4,
                "frames": sample["frames_a"],
            },
        ).json()
        sid_local = created["session_id"]
        client.post(
            "/api/advance",
            json={"session_id": sid_local, "to_frame": len(sample["frames_a"])},
        )
        response = client.post(
            "/api/compare",
            json={
                "session_id": sid_local,
                "start_frame": 0,
                "end_frame": len(sample["frames_a"]),
                "frames_b": sample["frames_b"],
            },
        )
        assert response.status_code == 200
        assert response.json()["status"] == expected, name


def test_late_compare_response_cannot_clobber_newer_one(client, sid):
    """The endpoint is read-only; request_id discipline is enforced client-side.

    Here we verify the server simply echoes the request_id unchanged so the UI
    can discard stale replies, and that two rapid calls both succeed without
    mutating the session.
    """
    first = client.post(
        "/api/compare",
        json={
            "session_id": sid,
            "start_frame": 0,
            "end_frame": 12,
            "frames_b": good_b(),
            "request_id": "older",
        },
    )
    second = client.post(
        "/api/compare",
        json={
            "session_id": sid,
            "start_frame": 0,
            "end_frame": 12,
            "frames_b": good_b(),
            "request_id": "newer",
        },
    )
    assert first.json()["request_id"] == "older"
    assert second.json()["request_id"] == "newer"
    snapshot = client.get(f"/api/sessions/{sid}").json()
    assert snapshot["next_frame"] == 16
