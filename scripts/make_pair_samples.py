#!/usr/bin/env python3
"""Generate bundled two-segment comparison demos under ``samples/pairs``.

Each demo contains two 8-64 frame segments over the same vocabulary plus the
frame range the UI should preselect as segment A.  They cover the two
situations the duplicate-recording check must distinguish:

``pair_duplicate.json``
    near-duplicate takes with the same pronunciation events at slightly
    different local speeds (every speed ratio stays within 2x), so the panel
    reports a high similarity and no conflict;
``pair_conflict.json``
    recordings that look superficially similar but contain a different event
    in the middle (and a skipped event), so the panel must refuse to call them
    duplicates and list the conflicts instead.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "samples" / "pairs"

VOCAB = ["<blank>", "a", "b", "c"]


def row(blank=0.0, a=0.0, b=0.0, c=0.0):
    return [blank, a, b, c]


BLANK = row(blank=0.94, a=0.02, b=0.02, c=0.02)
A_HI = row(blank=0.05, a=0.90, b=0.03, c=0.02)
B_HI = row(blank=0.05, a=0.04, b=0.88, c=0.03)
C_HI = row(blank=0.08, a=0.04, b=0.03, c=0.85)
# Nothing reaches 0.8 here and the blank is not argmax: an ambiguous frame.
AMBIG = row(blank=0.30, a=0.36, b=0.20, c=0.14)


def normalise(rows):
    out = []
    for r in rows:
        total = sum(r)
        out.append([round(v / total, 4) for v in r] if total > 0 else list(r))
    return out


def duplicate_demo():
    # Segment A: a (2 frames) - blank - b (2) - blank - c (1), padded to 12.
    seg_a = [
        BLANK,
        A_HI, A_HI,
        BLANK,
        B_HI, B_HI,
        BLANK,
        C_HI,
        BLANK, BLANK, BLANK, BLANK,
    ]
    # Segment B: same events, local tempo changes (a stretches 2 -> 3 frames,
    # b shrinks 2 -> 1), no ratio above 2.  14 frames total.
    seg_b = [
        BLANK, BLANK,
        A_HI, A_HI, A_HI,
        BLANK,
        B_HI,
        BLANK,
        C_HI, C_HI,
        BLANK, BLANK, BLANK, BLANK,
    ]
    assert 8 <= len(seg_a) <= 64 and 8 <= len(seg_b) <= 64
    return {
        "title": "疑似重复采集（节奏略变）",
        "description": (
            "两段录音的高置信发音事件完全对应，局部速度均不超过 2 倍，"
            "应给出高相似度且无冲突；大量 blank 帧不计入重复证据。"
        ),
        "vocab": VOCAB,
        "suggested_range": [0, len(seg_a)],
        "frames_a": normalise(seg_a),
        "frames_b": normalise(seg_b),
    }


def conflict_demo():
    # Segment A: a - blank - b - blank - c.
    seg_a = [
        BLANK,
        A_HI, A_HI,
        BLANK,
        B_HI, B_HI,
        BLANK,
        C_HI,
        BLANK, BLANK, BLANK, BLANK,
    ]
    # Segment B: a is present, but the middle event is c instead of b, and the
    # final c has no counterpart event on A's timeline ordering; the alignment
    # must surface label / skipped-event conflicts rather than similarity.
    seg_b = [
        BLANK,
        A_HI, A_HI, A_HI,
        BLANK,
        C_HI, C_HI,
        BLANK,
        AMBIG,
        BLANK, BLANK, BLANK, BLANK,
    ]
    assert 8 <= len(seg_a) <= 64 and 8 <= len(seg_b) <= 64
    return {
        "title": "同位置不同发音（冲突）",
        "description": (
            "中段事件发音不同且存在无法对应的高置信事件，必须提示冲突，"
            "不能因为两段都有大量 blank 就判为重复。"
        ),
        "vocab": VOCAB,
        "suggested_range": [0, len(seg_a)],
        "frames_a": normalise(seg_a),
        "frames_b": normalise(seg_b),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, payload in (
        ("pair_duplicate.json", duplicate_demo()),
        ("pair_conflict.json", conflict_demo()),
    ):
        path = OUT / name
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"wrote {path.relative_to(ROOT)} "
            f"(A={len(payload['frames_a'])} frames, B={len(payload['frames_b'])} frames)"
        )


if __name__ == "__main__":
    main()
