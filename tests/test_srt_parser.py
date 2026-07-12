"""Unit tests for the SRT subtitle parser in coding/code/data_preocessing.py.

Run from the repo root:
    .venv\\Scripts\\python.exe -m pytest -v

A unit test feeds a function a KNOWN input and checks it produces the EXPECTED
output. If someone later changes the parser and breaks it, these tests fail
loudly instead of silently corrupting the dataset.
"""
import sys
from pathlib import Path

import pytest

# The parser lives in coding/code/, which is not an installable package, so we
# add that folder to Python's import path before importing it.
# parents[1] = repo root (this file is at <repo>/tests/test_srt_parser.py).
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "coding" / "code"))

from data_preocessing import parse_srt_excluding_common, read_text_with_fallback  # noqa: E402


# A tiny, fake subtitle file whose correct answers we know by hand.
SAMPLE_SRT = """1
00:00:01,000 --> 00:00:04,000
Hello world

2
00:00:05,000 --> 00:01:30,500
This is a test
"""


def test_word_count(tmp_path):
    # tmp_path is a fresh temporary folder pytest gives us for each test.
    srt = tmp_path / "sample.srt"
    srt.write_text(SAMPLE_SRT, encoding="utf-8")

    word_count, _minutes, _top_words, _content = parse_srt_excluding_common(srt)

    # "Hello world" + "This is a test" = 6 words
    assert word_count == 6


def test_runtime_from_last_timestamp(tmp_path):
    srt = tmp_path / "sample.srt"
    srt.write_text(SAMPLE_SRT, encoding="utf-8")

    _wc, minutes, _tw, _c = parse_srt_excluding_common(srt)

    # Last timestamp 00:01:30,500 -> 1 min + 30s + 500ms = 1.5083... minutes
    assert minutes == pytest.approx(1.5083, abs=0.001)


def test_handles_non_utf8_without_crashing(tmp_path):
    # 0xE9 is 'e-acute' in cp1252/latin-1 but is INVALID standalone UTF-8.
    # The fallback reader must return text, not raise UnicodeDecodeError.
    srt = tmp_path / "weird.srt"
    srt.write_bytes(b"1\n00:00:01,000 --> 00:00:02,000\ncaf\xe9 scene\n")

    text = read_text_with_fallback(srt)

    assert isinstance(text, str)
    assert "caf" in text


def test_empty_file_yields_zero_words(tmp_path):
    srt = tmp_path / "empty.srt"
    srt.write_text("", encoding="utf-8")

    word_count, minutes, _tw, _c = parse_srt_excluding_common(srt)

    assert word_count == 0
    assert minutes == 0
