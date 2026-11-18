"""Numerical core for the two-segment duplicate-recording check.

The quality reviewer's question is narrow: could two recordings be the same
collection repeated?  Blank frames must never count as evidence of repetition,
so the whole comparison is built on *pronunciation events* — high-confidence
non-blank frames — rather than on raw frame similarity.

Pipeline
--------
1. :func:`normalize_frames` divides every frame by its own sum.  A frame whose
   sum is zero cannot be normalised; the caller keeps its previous result and
   surfaces the failure instead of producing NaNs.
2. :func:`extract_events` marks a frame as a pronunciation frame only when a
   non-blank label reaches ``EVENT_THRESHOLD`` (0.8).  Consecutive frames of the
   same label form one event; two runs of the same label count as two events
   only when at least one *blank-dominant* frame (argmax is the blank class)
   sits between them.  Frames where nothing reaches the threshold are merely
   ambiguous and never split an event.
3. :func:`align_frames` runs monotonic DTW whose only moves are (1,0), (0,1)
   and (1,1), restricted so that every prefix of the warp path stays inside a
   slope-2 band: neither side may run faster than 2x locally.  Event frames
   aligned with the wrong label are heavily penalised, so skipping a
   high-confidence event cannot hide inside a cheap blank-to-blank path.
4. :func:`compare_segments` projects the warp path onto the events, requires a
   one-to-one same-label correspondence, flags tempo violations per event and
   turns the result into a JSON-safe dict with a 0..1 similarity that is built
   only from event evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

BLANK_INDEX = 0
EVENT_THRESHOLD = 0.8
MIN_SEGMENT_FRAMES = 8
MAX_SEGMENT_FRAMES = 64

# DTW cell costs.  A wrong-label event match is the most expensive move;
# stretching an event over a quiet frame is allowed but discouraged.  Two
# non-event frames align for free, which is what keeps blanks from acting as
# positive repetition evidence.
COST_LABEL_MISMATCH = 1.0
COST_EVENT_VS_BLANK = 0.6
COST_EVENT_VS_AMBIGUOUS = 0.4

# Similarity weights.  Label agreement dominates; timing (half local tempo from
# frame-count ratios, half event-centre position on a normalised timeline) and
# the retained confidence of the events fill the rest.  The sum is 1.0.
W_LABEL = 0.6
W_TIMING = 0.3
W_CONFIDENCE = 0.1


class CompareError(ValueError):
    """Invalid segment data or a frame that cannot be normalised."""


@dataclass
class Event:
    """One pronunciation event: a run of high-confidence frames of one label."""

    index: int
    label: int
    token: str
    frames: List[int]
    peak: float

    @property
    def start(self) -> int:
        return self.frames[0]

    @property
    def end(self) -> int:
        return self.frames[-1]

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    def to_dict(self) -> dict:
        return {
            "event": self.index,
            "label": self.label,
            "token": self.token,
            "start": self.start,
            "end": self.end,
            "frame_count": self.frame_count,
            "peak": round(self.peak, 4),
            "frames": list(self.frames),
        }


# ---------------------------------------------------------------------------
# 1. Per-frame normalisation
# ---------------------------------------------------------------------------


def normalize_frames(probs) -> np.ndarray:
    """L1-normalise every frame independently.

    A frame that sums to zero has no defined conditional distribution; raising
    here lets the API preserve the previous comparison result instead of
    emitting NaNs.
    """
    arr = np.asarray(probs, dtype=np.float64)
    if arr.ndim != 2 or arr.size == 0:
        raise CompareError("帧数据必须是非空的二维数组")
    if np.isnan(arr).any() or np.isinf(arr).any():
        raise CompareError("帧概率包含 NaN 或无穷值，无法归一化")
    if np.any(arr < 0.0):
        raise CompareError("帧概率不能为负数")
    sums = arr.sum(axis=1)
    if np.any(sums <= 0.0):
        raise CompareError("存在全零帧，无法逐帧归一化；已保留上次对读结果")
    return arr / sums[:, None]


# ---------------------------------------------------------------------------
# 2. Event extraction
# ---------------------------------------------------------------------------


def extract_events(
    norm: np.ndarray, vocab: Sequence[str]
) -> Tuple[List[str], List[int], List[Event]]:
    """Return ``(frame_kinds, event_label_per_frame, events)``.

    ``frame_kinds[t]`` is ``"event"`` (a non-blank label reaches 0.8),
    ``"blank"`` (the blank class is argmax) or ``"ambiguous"`` (nothing reaches
    the threshold).  ``event_label_per_frame[t]`` is the label index on event
    frames and ``-1`` otherwise.
    """
    n_frames, vocab_size = norm.shape
    if vocab_size != len(vocab):
        raise CompareError("两段概率的词表宽度必须与会话词表一致")

    kinds: List[str] = ["ambiguous"] * n_frames
    event_labels: List[int] = [-1] * n_frames
    argmax = np.argmax(norm, axis=1)
    for t in range(n_frames):
        label = int(argmax[t])
        top = float(norm[t, label])
        if label == BLANK_INDEX:
            kinds[t] = "blank"
        elif top >= EVENT_THRESHOLD:
            kinds[t] = "event"
            event_labels[t] = label
        else:
            kinds[t] = "ambiguous"

    events: List[Event] = []
    open_label: Optional[int] = None
    open_frames: List[int] = []
    blank_seen = False

    def flush() -> None:
        if open_label is None:
            return
        peak = max(float(norm[t, open_label]) for t in open_frames)
        events.append(
            Event(
                index=len(events),
                label=open_label,
                token=vocab[open_label],
                frames=list(open_frames),
                peak=peak,
            )
        )

    for t in range(n_frames):
        kind = kinds[t]
        if kind == "event":
            label = event_labels[t]
            if open_label is None:
                open_label, open_frames, blank_seen = label, [t], False
            elif label == open_label and not blank_seen:
                # Same label with no blank-dominant frame in between: the run
                # collapses into a single event.
                open_frames.append(t)
            else:
                # Either a different label, or a blank split two same-label
                # runs into two genuinely separate pronunciations.
                flush()
                open_label, open_frames, blank_seen = label, [t], False
        elif kind == "blank":
            # Only a blank-dominant frame can split same-label events; an
            # ambiguous non-event frame cannot.
            if open_label is not None:
                blank_seen = True

    flush()
    return kinds, event_labels, events


# ---------------------------------------------------------------------------
# 3. Monotonic DTW with a slope-2 band
# ---------------------------------------------------------------------------


def _cell_cost(
    i: int,
    j: int,
    kinds_a: Sequence[str],
    kinds_b: Sequence[str],
    labels_a: Sequence[int],
    labels_b: Sequence[int],
) -> float:
    ea = labels_a[i]
    eb = labels_b[j]
    if ea >= 0 and eb >= 0:
        return 0.0 if ea == eb else COST_LABEL_MISMATCH
    if ea >= 0 or eb >= 0:
        # One side is a pronunciation frame, the other is not.  Stretching over
        # a blank-dominant frame is dearer than stretching over an ambiguous
        # one, but neither is free.
        other_kind = kinds_b[j] if ea >= 0 else kinds_a[i]
        return COST_EVENT_VS_BLANK if other_kind == "blank" else COST_EVENT_VS_AMBIGUOUS
    return 0.0


def align_frames(
    kinds_a: Sequence[str],
    kinds_b: Sequence[str],
    labels_a: Sequence[int],
    labels_b: Sequence[int],
) -> Optional[List[Tuple[int, int]]]:
    """Monotonic frame alignment with local speed <= 2x.

    Returns the warp path as ``[(i, j), ...]`` from ``(0, 0)`` to
    ``(n-1, m-1)``, or ``None`` when no slope-2 path exists.  The allowed moves
    are diagonal (1,1), advance-A (1,0) and advance-B (0,1); a single-axis move
    may not follow another move on the same axis, which is the local Sakoe-Chiba
    slope-2 rule — between diagonal steps neither side can advance more than
    twice as fast.  Ties prefer the diagonal move, then advancing A, then B, so
    the result is deterministic.
    """
    n, m = len(kinds_a), len(kinds_b)
    inf = float("inf")
    dp: List[List[float]] = [[inf] * m for _ in range(n)]
    # move that produced each cell: 0 = diagonal, 1 = advance A, 2 = advance B
    back: List[List[int]] = [[-1] * m for _ in range(n)]
    dp[0][0] = _cell_cost(0, 0, kinds_a, kinds_b, labels_a, labels_b)

    def cost(i: int, j: int) -> float:
        return _cell_cost(i, j, kinds_a, kinds_b, labels_a, labels_b)

    for i in range(n):
        for j in range(m):
            if i == 0 and j == 0:
                continue
            best = inf
            choice = -1
            # Diagonal predecessor: always legal.
            if i > 0 and j > 0 and dp[i - 1][j - 1] < inf:
                candidate = dp[i - 1][j - 1] + cost(i, j)
                if candidate < best:
                    best, choice = candidate, 0
            # Advance A: legal only if the predecessor was not itself an
            # advance-A move (no two consecutive A-axis steps).
            if i > 0 and dp[i - 1][j] < inf and back[i - 1][j] != 1:
                candidate = dp[i - 1][j] + cost(i, j)
                if candidate < best:
                    best, choice = candidate, 1
            # Advance B: same rule on the B axis.
            if j > 0 and dp[i][j - 1] < inf and back[i][j - 1] != 2:
                candidate = dp[i][j - 1] + cost(i, j)
                if candidate < best:
                    best, choice = candidate, 2
            if choice >= 0:
                dp[i][j] = best
                back[i][j] = choice

    if dp[n - 1][m - 1] == inf:
        return None

    path: List[Tuple[int, int]] = []
    i, j = n - 1, m - 1
    while i > 0 or j > 0:
        path.append((i, j))
        move = back[i][j]
        if move == 0:
            i, j = i - 1, j - 1
        elif move == 1:
            i = i - 1
        else:
            j = j - 1
    path.append((0, 0))
    path.reverse()
    return path


# ---------------------------------------------------------------------------
# 4. Event correspondence, conflicts and similarity
# ---------------------------------------------------------------------------


def _frame_maps(path: Sequence[Tuple[int, int]]) -> Tuple[Dict[int, Set[int]], Dict[int, Set[int]]]:
    map_a: Dict[int, Set[int]] = {}
    map_b: Dict[int, Set[int]] = {}
    for i, j in path:
        map_a.setdefault(i, set()).add(j)
        map_b.setdefault(j, set()).add(i)
    return map_a, map_b


def _opposite_events(
    event: Event,
    forward_map: Dict[int, Set[int]],
    opposite: Sequence[Event],
) -> List[int]:
    """Indices of opposite-side events touched through the warp path."""
    reached: Set[int] = set()
    for frame in event.frames:
        reached.update(forward_map.get(frame, set()))
    hits = []
    for other in opposite:
        if reached.intersection(other.frames):
            hits.append(other.index)
    return hits


def _speed_ratio(count_a: int, count_b: int) -> float:
    if count_a <= 0 or count_b <= 0:  # pragma: no cover - events always have frames
        return float("inf")
    return max(count_a, count_b) / min(count_a, count_b)


def compare_segments(probs_a, probs_b, vocab: Sequence[str]) -> dict:
    """Run the full duplicate-check pipeline and return a JSON-safe result."""
    vocab = list(vocab)
    for name, probs in (("A", probs_a), ("B", probs_b)):
        arr = np.asarray(probs, dtype=np.float64)
        if arr.ndim != 2:
            raise CompareError(f"{name} 段帧数据必须是二维数组")
        if not (MIN_SEGMENT_FRAMES <= arr.shape[0] <= MAX_SEGMENT_FRAMES):
            raise CompareError(
                f"{name} 段必须包含 {MIN_SEGMENT_FRAMES}～{MAX_SEGMENT_FRAMES} 帧"
            )
        if arr.shape[1] != len(vocab):
            raise CompareError(f"{name} 段每帧宽度必须等于词表大小 {len(vocab)}")

    norm_a = normalize_frames(probs_a)
    norm_b = normalize_frames(probs_b)
    kinds_a, labels_a, events_a = extract_events(norm_a, vocab)
    kinds_b, labels_b, events_b = extract_events(norm_b, vocab)

    conflicts: List[dict] = []
    pairs: List[dict] = []
    alignment: List[dict] = []
    status = "ok"
    reason: Optional[str] = None

    # No pronunciation events on either side: blanks are not evidence.
    if not events_a or not events_b:
        missing = []
        if not events_a:
            missing.append("A 段")
        if not events_b:
            missing.append("B 段")
        status = "insufficient"
        reason = (
            "、".join(missing)
            + "没有概率达到 0.8 的非 blank 发音事件，证据不足；blank 帧不能作为重复依据"
        )
        path = align_frames(kinds_a, kinds_b, labels_a, labels_b)
        if path is not None:
            alignment = [{"a": i, "b": j} for i, j in path]
        return _result(
            status=status,
            reason=reason,
            similarity=None,
            score_parts=None,
            events_a=events_a,
            events_b=events_b,
            pairs=pairs,
            conflicts=conflicts,
            alignment=alignment,
            kinds_a=kinds_a,
            kinds_b=kinds_b,
            norm_a=norm_a,
            norm_b=norm_b,
        )

    path = align_frames(kinds_a, kinds_b, labels_a, labels_b)
    if path is None:
        status = "conflict"
        conflicts.append(
            {
                "type": "unalignable",
                "message": (
                    f"两段长度为 {len(kinds_a)} / {len(kinds_b)} 帧，比例超过 2 倍，"
                    "不存在满足局部速度限制的单调对齐"
                ),
            }
        )
        return _result(
            status=status,
            reason="速度超限，事件无法对应",
            similarity=None,
            score_parts=None,
            events_a=events_a,
            events_b=events_b,
            pairs=pairs,
            conflicts=conflicts,
            alignment=alignment,
            kinds_a=kinds_a,
            kinds_b=kinds_b,
            norm_a=norm_a,
            norm_b=norm_b,
        )

    alignment = [{"a": i, "b": j} for i, j in path]
    map_a, map_b = _frame_maps(path)

    paired_b: Set[int] = set()

    # A -> B correspondence.
    for event in events_a:
        hits = _opposite_events(event, map_a, events_b)
        if not hits:
            status = "conflict"
            conflicts.append(
                {
                    "type": "skipped_event",
                    "side": "a",
                    "token": event.token,
                    "start": event.start,
                    "end": event.end,
                    "message": (
                        f"高置信事件被跳过：A 段「{event.token}」（帧 {event.start}-{event.end}）"
                        "在 B 段找不到对应发音事件"
                    ),
                }
            )
            continue
        if len(hits) > 1:
            status = "conflict"
            conflicts.append(
                {
                    "type": "one_to_many",
                    "side": "a",
                    "token": event.token,
                    "start": event.start,
                    "end": event.end,
                    "opposite_events": hits,
                    "message": (
                        f"事件对应不唯一：A 段「{event.token}」（帧 {event.start}-{event.end}）"
                        f"同时对应 B 段 {len(hits)} 个事件"
                    ),
                }
            )
            continue
        other = events_b[hits[0]]
        if other.label != event.label:
            status = "conflict"
            conflicts.append(
                {
                    "type": "label_mismatch",
                    "side": "a",
                    "token": event.token,
                    "opposite_token": other.token,
                    "start": event.start,
                    "end": event.end,
                    "opposite_start": other.start,
                    "opposite_end": other.end,
                    "message": (
                        f"发音冲突：A 段「{event.token}」（帧 {event.start}-{event.end}）对应 "
                        f"B 段「{other.token}」（帧 {other.start}-{other.end}）"
                    ),
                }
            )
            continue
        if other.index not in paired_b:
            paired_b.add(other.index)
            pairs.append(_pair_dict(event, other))

    # B -> A reverse check catches events that only B would report as skipped.
    for event in events_b:
        hits = _opposite_events(event, map_b, events_a)
        if not hits:
            status = "conflict"
            conflicts.append(
                {
                    "type": "skipped_event",
                    "side": "b",
                    "token": event.token,
                    "start": event.start,
                    "end": event.end,
                    "message": (
                        f"高置信事件被跳过：B 段「{event.token}」（帧 {event.start}-{event.end}）"
                        "在 A 段找不到对应发音事件"
                    ),
                }
            )
            continue
        if len(hits) > 1:
            status = "conflict"
            conflicts.append(
                {
                    "type": "one_to_many",
                    "side": "b",
                    "token": event.token,
                    "start": event.start,
                    "end": event.end,
                    "opposite_events": hits,
                    "message": (
                        f"事件对应不唯一：B 段「{event.token}」（帧 {event.start}-{event.end}）"
                        f"同时对应 A 段 {len(hits)} 个事件"
                    ),
                }
            )
            continue
        other = events_a[hits[0]]
        if other.label != event.label:
            already = any(
                c["type"] == "label_mismatch"
                and c.get("opposite_token") == event.token
                and c.get("token") == other.token
                for c in conflicts
            )
            if not already:
                status = "conflict"
                conflicts.append(
                    {
                        "type": "label_mismatch",
                        "side": "b",
                        "token": event.token,
                        "opposite_token": other.token,
                        "start": event.start,
                        "end": event.end,
                        "opposite_start": other.start,
                        "opposite_end": other.end,
                        "message": (
                            f"发音冲突：B 段「{event.token}」（帧 {event.start}-{event.end}）对应 "
                            f"A 段「{other.token}」（帧 {other.start}-{other.end}）"
                        ),
                    }
                )

    # Per-event local tempo check on the one-to-one same-label pairs.
    good_pairs = []
    for pair in pairs:
        ratio = _speed_ratio(pair["a_frame_count"], pair["b_frame_count"])
        pair["speed_ratio"] = round(ratio, 3)
        pair["tempo_ok"] = ratio <= 2.0 + 1e-9
        if not pair["tempo_ok"]:
            status = "conflict"
            conflicts.append(
                {
                    "type": "tempo",
                    "token": pair["token"],
                    "speed_ratio": pair["speed_ratio"],
                    "message": (
                        f"局部速度超过 2 倍：事件「{pair['token']}」对应帧跨度 "
                        f"{pair['a_frame_count']} ↔ {pair['b_frame_count']}"
                    ),
                }
            )
        else:
            good_pairs.append(pair)

    total_events = max(len(events_a), len(events_b), 1)
    label_score = len(good_pairs) / total_events
    n_a, n_b = len(kinds_a), len(kinds_b)
    if good_pairs:
        tempo_terms = [1.0 - (p["speed_ratio"] - 1.0) for p in good_pairs]
        # Position term: how far the event centres sit from each other once
        # frame indices are scaled to a common 0..1 timeline, so a recording
        # whose events drift in time scores lower even with identical durations.
        position_terms = [
            1.0
            - abs(
                (p["a_start"] + p["a_end"]) / (2 * n_a)
                - (p["b_start"] + p["b_end"]) / (2 * n_b)
            )
            for p in good_pairs
        ]
        timing_score = float(
            0.5 * np.mean(tempo_terms) + 0.5 * np.mean(position_terms)
        )
    else:
        timing_score = 0.0
    timing_score = min(1.0, max(0.0, timing_score))
    all_peaks = [e.peak for e in events_a] + [e.peak for e in events_b]
    confidence_score = float(np.mean(all_peaks)) if all_peaks else 0.0
    similarity = (
        W_LABEL * label_score
        + W_TIMING * timing_score
        + W_CONFIDENCE * confidence_score
    )
    similarity = round(min(1.0, max(0.0, similarity)), 4)
    score_parts = {
        "label": round(label_score, 4),
        "timing": round(timing_score, 4),
        "confidence": round(confidence_score, 4),
        "matched_events": len(good_pairs),
        "total_events": total_events,
    }

    if status == "conflict":
        reason = "存在无法对应或速度超限的发音事件，详见冲突列表"
    else:
        reason = None

    return _result(
        status=status,
        reason=reason,
        similarity=similarity,
        score_parts=score_parts,
        events_a=events_a,
        events_b=events_b,
        pairs=pairs,
        conflicts=conflicts,
        alignment=alignment,
        kinds_a=kinds_a,
        kinds_b=kinds_b,
        norm_a=norm_a,
        norm_b=norm_b,
    )


def _pair_dict(event_a: Event, event_b: Event) -> dict:
    return {
        "a_event": event_a.index,
        "b_event": event_b.index,
        "label": event_a.label,
        "token": event_a.token,
        "a_start": event_a.start,
        "a_end": event_a.end,
        "b_start": event_b.start,
        "b_end": event_b.end,
        "a_frame_count": event_a.frame_count,
        "b_frame_count": event_b.frame_count,
        "a_peak": round(event_a.peak, 4),
        "b_peak": round(event_b.peak, 4),
        "speed_ratio": round(
            _speed_ratio(event_a.frame_count, event_b.frame_count), 3
        ),
        "label_match": event_a.label == event_b.label,
        "tempo_ok": True,
    }


def _result(
    *,
    status: str,
    reason: Optional[str],
    similarity: Optional[float],
    score_parts: Optional[dict],
    events_a: Sequence[Event],
    events_b: Sequence[Event],
    pairs: Sequence[dict],
    conflicts: Sequence[dict],
    alignment: Sequence[dict],
    kinds_a: Sequence[str],
    kinds_b: Sequence[str],
    norm_a: np.ndarray,
    norm_b: np.ndarray,
) -> dict:
    return {
        "status": status,
        "reason": reason,
        "similarity": similarity,
        "score_parts": score_parts,
        "events_a": [e.to_dict() for e in events_a],
        "events_b": [e.to_dict() for e in events_b],
        "pairs": list(pairs),
        "conflicts": list(conflicts),
        "alignment": list(alignment),
        "frame_kinds_a": list(kinds_a),
        "frame_kinds_b": list(kinds_b),
        "lengths": {"a": int(norm_a.shape[0]), "b": int(norm_b.shape[0])},
    }
