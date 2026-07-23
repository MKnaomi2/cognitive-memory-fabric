from pathlib import Path

import pytest

from hippocampal_memory.telemetry import FrameRecorder, read_recording, serve


def test_recording_round_trip_and_truncation_guard(tmp_path: Path) -> None:
    target = tmp_path / "sample.hmrec"
    with FrameRecorder(target) as recorder:
        recorder.append(b"one")
        recorder.append(b"two")
    assert list(read_recording(target)) == [b"one", b"two"]
    target.write_bytes(target.read_bytes()[:-1])
    with pytest.raises(ValueError):
        list(read_recording(target))


def test_observatory_refuses_non_loopback_binding() -> None:
    with pytest.raises(ValueError):
        serve(object(), host="0.0.0.0")
