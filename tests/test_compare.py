"""Unit tests for the two-segment duplicate-check core (no HTTP)."""

import numpy as np
import pytest

from app.compare import (
    CompareError,
    align_frames,
    compare_segments,
    extract_events,
    normalize_frames,
)

VOCAB = ["<blank>", "a", "b", "c"]


def row(blank=0.0, a=0.0, b=0.0, c=0.0):
    return [blank, a, b, c]


BLANK = row(blank=0.94, a=0.02, b=0.02, c=0.02)
A_HI = row(blank=0.05, a=0.90, b=0.03, c=0.02)
B_HI = row(blank=0.05, a=0.04, b=0.88, c=0.03)
C_HI = row(blank=0.08, a=0.04, b=0.03, c=0.85)
AMBIG = row(blank=0.30, a=0.36, b=0.20, c=0.14)


def pad(rows, n):
    """Pad to at least n frames with blanks (segment length must be 8-64)."""
    out = list(rows)
    while len(out) < n:
        out.append(BLANK)
    return out


# ------------------------------------------------------------- normalisation


def test_normalize_divides_each_frame_by_its_own_sum():
    frames = [[2.0, 2.0, 0.0, 0.0], [0.0, 3.0, 3.0, 0.0]]
    norm = normalize_frames(frames)
    assert np.allclose(norm.sum(axis=1), 1.0)
    assert np.allclose(norm[0], [0.5, 0.5, 0.0, 0.0])
    assert np.allclose(norm[1], [0.0, 0.5, 0.5, 0.0])


def test_all_zero_frame_raises_so_caller_keeps_previous_result():
    frames = pad([[0.0, 0.0, 0.0, 0.0]], 8)
    with pytest.raises(CompareError, match="全零帧"):
        normalize_frames(frames)


def test_negative_and_nan_frames_are_rejected():
    with pytest.raises(CompareError):
        normalize_frames(pad([[-0.1, 0.5, 0.3, 0.3]], 8))
    with pytest.raises(CompareError):
        normalize_frames(pad([[float("nan"), 0.5, 0.3, 0.2]], 8))


# ------------------------------------------------------------- event extraction


def test_event_requires_non_blank_probability_at_least_0_8():
    weak = pad([row(blank=0.15, a=0.79, b=0.03, c=0.03)], 8)
    _, _, events = extract_events(normalize_frames(weak), VOCAB)
    assert events == []

    strong = pad([row(blank=0.15, a=0.80, b=0.03, c=0.02)], 8)
    kinds, _, events = extract_events(normalize_frames(strong), VOCAB)
    assert len(events) == 1 and events[0].token == "a"
    assert kinds[0] == "event"


def test_consecutive_same_label_collapses_without_blank_between():
    frames = pad([*[A_HI] * 4, AMBIG, A_HI], 8)
    kinds, _, events = extract_events(normalize_frames(frames), VOCAB)
    # The ambiguous frame has no blank argmax, so it cannot split the run.
    assert len(events) == 1
    assert events[0].frame_count == 5
    assert "ambiguous" in kinds


def test_blank_dominant_frame_splits_same_label_into_two_events():
    frames = pad([*[A_HI] * 2, BLANK, *[A_HI] * 2], 8)
    _, _, events = extract_events(normalize_frames(frames), VOCAB)
    assert len(events) == 2
    assert all(e.token == "a" for e in events)
    assert events[0].frames == [0, 1]
    assert events[1].frames == [2 + 1, 2 + 2]


def test_different_labels_are_separate_events_without_blank():
    frames = pad([*[A_HI] * 2, *[B_HI] * 2], 8)
    _, _, events = extract_events(normalize_frames(frames), VOCAB)
    assert [e.token for e in events] == ["a", "b"]


def test_blank_argmax_frame_is_classified_blank():
    kinds, _, _ = extract_events(normalize_frames(pad([BLANK], 8)), VOCAB)
    assert kinds[0] == "blank"


# ------------------------------------------------------------- DTW alignment


def test_dtw_path_is_monotonic_and_covers_both_ends():
    kinds_a = ["blank", "event", "event", "blank", "blank", "blank", "blank", "blank"]
    labels_a = [-1, 1, 1, -1, -1, -1, -1, -1]
    kinds_b = ["blank", "blank", "event", "event", "event", "blank", "blank", "blank"]
    labels_b = [-1, -1, 1, 1, 1, -1, -1, -1]
    path = align_frames(kinds_a, kinds_b, labels_a, labels_b)
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (7, 7)
    for (i1, j1), (i2, j2) in zip(path, path[1:]):
        assert 0 <= i2 - i1 <= 1 and 0 <= j2 - j1 <= 1
        assert (i2, j2) != (i1, j1)


def test_dtw_forbids_two_consecutive_same_axis_moves():
    path = align_frames(["blank"] * 4, ["blank"] * 2, [-1] * 4, [-1] * 2)
    assert path is not None and path[-1] == (3, 1)
    for k in range(1, len(path) - 1):
        d1 = (path[k][0] - path[k - 1][0], path[k][1] - path[k - 1][1])
        d2 = (path[k + 1][0] - path[k][0], path[k + 1][1] - path[k][1])
        assert not (d1 == (1, 0) and d2 == (1, 0))
        assert not (d1 == (0, 1) and d2 == (0, 1))


def test_dtw_returns_none_when_length_ratio_exceeds_two():
    assert align_frames(["blank"] * 5, ["blank"] * 2, [-1] * 5, [-1] * 2) is None
    assert align_frames(["blank"] * 64, ["blank"] * 8, [-1] * 64, [-1] * 8) is None


def test_dtw_ratio_exactly_two_is_feasible():
    path = align_frames(["blank"] * 64, ["blank"] * 32, [-1] * 64, [-1] * 32)
    assert path is not None and path[-1] == (63, 31)


def test_dtw_prefers_matching_over_skipping_high_confidence_event():
    # Same label on both sides must align event-to-event rather than detour.
    kinds_a = ["blank", "blank", "event", "blank", "blank", "blank", "blank", "blank"]
    labels_a = [-1, -1, 1, -1, -1, -1, -1, -1]
    kinds_b = ["blank", "blank", "event", "blank", "blank", "blank", "blank", "blank"]
    labels_b = [-1, -1, 1, -1, -1, -1, -1, -1]
    path = align_frames(kinds_a, kinds_b, labels_a, labels_b)
    assert (2, 2) in path


# ------------------------------------------------------------- full comparison


def test_duplicate_segments_get_high_similarity_and_no_conflict():
    seg_a = pad([BLANK, *[A_HI] * 2, BLANK, *[B_HI] * 2, BLANK, C_HI], 12)
    seg_b = pad([BLANK, *[A_HI] * 3, BLANK, B_HI, BLANK, *[C_HI] * 2], 14)
    result = compare_segments(seg_a, seg_b, VOCAB)
    assert result["status"] == "ok"
    assert result["conflicts"] == []
    assert 0.0 <= result["similarity"] <= 1.0
    assert result["similarity"] > 0.7
    assert [p["token"] for p in result["pairs"]] == ["a", "b", "c"]
    for pair in result["pairs"]:
        assert pair["label_match"] and pair["tempo_ok"]
        assert pair["speed_ratio"] <= 2.0 + 1e-9


def test_label_mismatch_is_a_conflict_not_a_similarity():
    seg_a = pad([BLANK, *[A_HI] * 2, BLANK, *[B_HI] * 2, BLANK, C_HI], 12)
    seg_b = pad([BLANK, *[A_HI] * 3, BLANK, *[C_HI] * 2, BLANK, AMBIG], 13)
    result = compare_segments(seg_a, seg_b, VOCAB)
    assert result["status"] == "conflict"
    kinds = {c["type"] for c in result["conflicts"]}
    assert "label_mismatch" in kinds or "skipped_event" in kinds


def test_skipped_high_confidence_event_is_conflict():
    seg_a = pad([BLANK, *[A_HI] * 2, BLANK, *[B_HI] * 2, BLANK, C_HI], 12)
    seg_b = pad([*[BLANK] * 4, *[A_HI] * 2, *[BLANK] * 2], 8)
    result = compare_segments(seg_a, seg_b, VOCAB)
    assert result["status"] == "conflict"
    assert any(c["type"] == "skipped_event" for c in result["conflicts"])


def test_no_events_on_either_side_is_insufficient_not_duplicate():
    seg_a = [BLANK] * 10
    seg_b = [BLANK] * 12
    result = compare_segments(seg_a, seg_b, VOCAB)
    assert result["status"] == "insufficient"
    assert result["similarity"] is None
    assert "证据不足" in result["reason"]
    # Blank-to-blank frames still produce an alignment path for inspection.
    assert result["alignment"]


def test_no_events_on_one_side_is_also_insufficient():
    seg_a = pad([BLANK, *[A_HI] * 2, BLANK, *[B_HI] * 2], 10)
    seg_b = [BLANK] * 10
    result = compare_segments(seg_a, seg_b, VOCAB)
    assert result["status"] == "insufficient"
    assert result["similarity"] is None


def test_many_blanks_do_not_inflate_similarity():
    # Both segments share one event surrounded by blanks.  Blank-to-blank
    # frames must not act as positive evidence: the similarity must be driven
    # entirely by event components, so an identical-timing pair beats a pair
    # whose event drifts even though both contain the same count of blanks, and
    # a blank-only comparison never produces a similarity at all.
    aligned_pair = (
        pad([*[BLANK] * 5, *[A_HI] * 2, BLANK], 8),
        pad([*[BLANK] * 5, *[A_HI] * 2, BLANK], 8),
    )
    drifted_pair = (
        pad([*[BLANK] * 5, *[A_HI] * 2, BLANK], 8),
        pad([*[BLANK] * 4, *[A_HI] * 2, *[BLANK] * 2], 8),
    )
    aligned = compare_segments(*aligned_pair, VOCAB)
    drifted = compare_segments(*drifted_pair, VOCAB)
    assert aligned["status"] == "ok" and aligned["score_parts"]["label"] == 1.0
    assert drifted["status"] == "ok"
    assert aligned["similarity"] > drifted["similarity"]

    blank_only = compare_segments([BLANK] * 8, [BLANK] * 8, VOCAB)
    assert blank_only["status"] == "insufficient"
    assert blank_only["similarity"] is None


def test_unalignable_length_ratio_is_conflict():
    seg_a = pad([*[BLANK] * 3, *[A_HI] * 2, *[BLANK] * 3], 8)
    seg_b = pad([*[BLANK] * 30, *[A_HI] * 4, *[BLANK] * 30], 64)
    result = compare_segments(seg_a, seg_b, VOCAB)
    assert result["status"] == "conflict"
    assert any(c["type"] == "unalignable" for c in result["conflicts"])
    assert result["similarity"] is None


def test_segment_length_bounds_enforced():
    good = [BLANK] * 8
    with pytest.raises(CompareError, match="8～64"):
        compare_segments([BLANK] * 7, good, VOCAB)
    with pytest.raises(CompareError, match="8～64"):
        compare_segments(good, [BLANK] * 65, VOCAB)


def test_vocab_width_mismatch_rejected():
    seg_a = pad([A_HI], 8)
    seg_b = [[0.5, 0.5, 0.0] for _ in range(8)]
    with pytest.raises(CompareError):
        compare_segments(seg_a, seg_b, VOCAB)


def test_all_zero_segment_raises_and_carries_preservation_hint():
    seg_a = pad([A_HI], 8)
    seg_b = [*[BLANK] * 7, [0.0, 0.0, 0.0, 0.0]]
    with pytest.raises(CompareError, match="保留上次对读结果"):
        compare_segments(seg_a, seg_b, VOCAB)


def test_result_is_json_safe():
    seg_a = pad([BLANK, *[A_HI] * 2, BLANK, *[B_HI] * 2], 10)
    seg_b = pad([BLANK, *[A_HI] * 3, BLANK, B_HI], 10)
    import json

    result = compare_segments(seg_a, seg_b, VOCAB)
    encoded = json.dumps(result, ensure_ascii=False)
    assert "NaN" not in encoded
    decoded = json.loads(encoded)
    assert decoded["lengths"] == {"a": 10, "b": 10}
