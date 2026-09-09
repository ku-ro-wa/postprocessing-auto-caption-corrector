"""Decision cache: span+model keyed, tolerant of a missing file, off entirely
when disabled."""

from __future__ import annotations

from pathlib import Path

from caption_checker.cache import CachedCorrection, DecisionCache


def test_set_then_get_round_trips(tmp_path: Path) -> None:
    cache = DecisionCache.load(tmp_path / "c.json")
    cache.set("sensus", "model-a", CachedCorrection("consensus", 0.9, "why"))

    got = cache.get("sensus", "model-a")
    assert got == CachedCorrection("consensus", 0.9, "why")


def test_missing_key_returns_none(tmp_path: Path) -> None:
    cache = DecisionCache.load(tmp_path / "c.json")
    cache.set("sensus", "model-a", CachedCorrection("consensus", 0.9))
    assert cache.get("other", "model-a") is None


def test_different_model_misses(tmp_path: Path) -> None:
    cache = DecisionCache.load(tmp_path / "c.json")
    cache.set("sensus", "model-a", CachedCorrection("consensus", 0.9))
    assert cache.get("sensus", "model-b") is None


def test_key_is_cleaned_span(tmp_path: Path) -> None:
    cache = DecisionCache.load(tmp_path / "c.json")
    cache.set("Sensus.", "m", CachedCorrection("consensus", 0.9))
    assert cache.get("sensus", "m") is not None


def test_missing_file_is_empty(tmp_path: Path) -> None:
    cache = DecisionCache.load(tmp_path / "nope.json")
    assert cache.get("anything", "m") is None


def test_persists_across_load(tmp_path: Path) -> None:
    path = tmp_path / "c.json"
    first = DecisionCache.load(path)
    first.set("sensus", "m", CachedCorrection("consensus", 0.8, "r"))
    first.save()

    second = DecisionCache.load(path)
    assert second.get("sensus", "m") == CachedCorrection("consensus", 0.8, "r")


def test_disabled_cache_never_reads_or_writes(tmp_path: Path) -> None:
    path = tmp_path / "c.json"
    cache = DecisionCache.load(path, enabled=False)
    cache.set("sensus", "m", CachedCorrection("consensus", 0.9))
    cache.save()

    assert cache.get("sensus", "m") is None
    assert not path.exists()


def test_null_replacement_round_trips(tmp_path: Path) -> None:
    cache = DecisionCache.load(tmp_path / "c.json")
    cache.set("lease", "m", CachedCorrection(None, 0.9, "not an error"))
    assert cache.get("lease", "m") == CachedCorrection(None, 0.9, "not an error")
