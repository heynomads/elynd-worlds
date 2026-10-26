"""Writes the Python timeline for a few sessions, for the Flutter parity test.

    python scripts/timeline_parity.py > flutter/elynd_worlds_scene/test/fixtures/timeline_parity.json
"""
import json

from elynd_worlds import timeline

CASES = [
    {"duration": 600, "silences": [[120, 180], [300, 320]], "cues": [], "sleep": False},
    {"duration": 300, "silences": [], "cues": [["inhale", 30], ["exhale", 34]], "sleep": False},
    {"duration": 1200, "silences": [[400, 470]], "cues": [], "sleep": True},
    {"duration": 95.5, "silences": [[10, 50]], "cues": [["inhale", 60]], "sleep": False},
]

out = []
for case in CASES:
    built = timeline.build(case["duration"], [timeline.Silence(*s) for s in case["silences"]],
                           [timeline.Cue(at, kind) for kind, at in case["cues"]], case["sleep"])
    out.append({**case, "breath": [f["breath"] for f in built["frames"]],
                "visibility": [f["visibility"] for f in built["frames"]]})
print(json.dumps(out))
