"""Tests for Hugging Face Hub dataset input (hf:// URIs)."""
import pytest

from sanitizer_pro import hub
from sanitizer_pro.hub import (
    HFDatasetRef, is_hub_uri, iter_hub_records, parse_hf_uri, resolve_parquet_urls,
)
from sanitizer_pro.utils import ConfigurationError


class TestParseUri:
    def test_repo_only(self):
        ref = parse_hf_uri("hf://openai/gsm8k")
        assert ref == HFDatasetRef(repo="openai/gsm8k", config=None, split="train")

    def test_repo_config(self):
        ref = parse_hf_uri("hf://openai/gsm8k/main")
        assert ref.config == "main" and ref.split == "train"

    def test_repo_config_split(self):
        ref = parse_hf_uri("hf://openai/gsm8k/main/test")
        assert ref == HFDatasetRef(repo="openai/gsm8k", config="main", split="test")

    def test_invalid_uris(self):
        for bad in ("hf://", "hf://onlyowner", "hf://a/b/c/d/e", "s3://a/b"):
            with pytest.raises(ConfigurationError):
                parse_hf_uri(bad)

    def test_is_hub_uri(self):
        assert is_hub_uri("hf://a/b")
        assert not is_hub_uri("data.jsonl")
        assert not is_hub_uri(None)


class TestResolveUrls:
    def _patch(self, monkeypatch, api_json):
        monkeypatch.setattr(hub, 'list_parquet', lambda repo: api_json)

    def test_explicit_config_and_split(self, monkeypatch):
        self._patch(monkeypatch, {"main": {"train": ["u1"], "test": ["u2"]}})
        assert resolve_parquet_urls(HFDatasetRef("o/n", "main", "test")) == ["u2"]

    def test_default_config_prefers_default(self, monkeypatch):
        self._patch(monkeypatch, {"zz": {"train": ["a"]}, "default": {"train": ["b"]}})
        assert resolve_parquet_urls(HFDatasetRef("o/n")) == ["b"]

    def test_default_config_falls_back_to_first_sorted(self, monkeypatch):
        self._patch(monkeypatch, {"beta": {"train": ["b"]}, "alpha": {"train": ["a"]}})
        assert resolve_parquet_urls(HFDatasetRef("o/n")) == ["a"]

    def test_missing_config(self, monkeypatch):
        self._patch(monkeypatch, {"main": {"train": ["u"]}})
        with pytest.raises(ConfigurationError, match="not found"):
            resolve_parquet_urls(HFDatasetRef("o/n", "nope"))

    def test_missing_split(self, monkeypatch):
        self._patch(monkeypatch, {"main": {"train": ["u"]}})
        with pytest.raises(ConfigurationError, match="Split 'test'"):
            resolve_parquet_urls(HFDatasetRef("o/n", "main", "test"))


def _pyarrow_available() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _pyarrow_available(), reason="pyarrow not installed")
class TestIterRecords:
    def test_streams_records_from_parquet(self, tmp_path, monkeypatch):
        import pyarrow as pa
        import pyarrow.parquet as pq
        shard = tmp_path / "part-000.parquet"
        pq.write_table(pa.Table.from_pylist(
            [{"text": "row one"}, {"text": "row two"}]), shard)
        monkeypatch.setattr(hub, 'download_parquet', lambda ref, cache_dir=None: [shard])
        records = list(iter_hub_records("hf://o/n/main/train"))
        assert records == [{"text": "row one"}, {"text": "row two"}]
