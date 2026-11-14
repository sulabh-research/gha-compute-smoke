"""Offline tests for the builder: local parquet source and LocalSink only (no network, no tokens)."""
import hashlib
import json
from pathlib import Path
import sys

import httpx2
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError, RemoteEntryNotFoundError
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from builder import inputs, run  # noqa: E402

ROWS = 230


def make_cfg(**overrides):
    cfg = {"source": {"repo_id": "owner/name", "revision": "a" * 40, "files": ["train.parquet"], "license": "mit"},
           "text_column": "text", "keep_columns": ["label"], "num_shards": 3, "checkpoint_rows": 20}
    cfg.update(overrides)
    return run.validate_config(cfg)


@pytest.fixture
def source_file(tmp_path):
    path = tmp_path / "train.parquet"
    table = pa.table({"text": [f"  Row  NUMBER {i}\tÉtude " for i in range(ROWS)], "label": list(range(ROWS))})
    pq.write_table(table, path, row_group_size=50)  # several row groups
    return path


class CountingSource(run.ParquetSource):
    def __init__(self, path):
        super().__init__(["train.parquet"], lambda _: open(path, "rb"))
        self.reads = []

    def read(self, start, stop, columns):
        self.reads.append((start, stop))
        return super().read(start, stop, columns)


class FailingSink(run.LocalSink):
    """Fails on the commit after `ok` successful commits (failure injection lives in tests only)."""

    def __init__(self, root, ok):
        super().__init__(root)
        self.ok = ok

    def commit(self, files, message):
        if self.ok == 0:
            raise RuntimeError("injected failure")
        self.ok -= 1
        super().commit(files, message)


def tree(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(root).rglob("*")) if p.is_file()}


def test_shard_ranges_tile_rows_without_overlap():
    for total, shards in ((230, 3), (7, 7), (1000, 256)):
        ranges = [run.shard_range(total, shards, k) for k in range(shards)]
        assert ranges[0][0] == 0 and ranges[-1][1] == total
        assert all(a[1] == b[0] for a, b in zip(ranges, ranges[1:]))
    with pytest.raises(ValueError):
        run.shard_range(2, 3, 0)  # an empty shard is a config error
    with pytest.raises(ValueError):
        run.shard_range(10, 3, 3)


def test_transform_is_deterministic_and_normalizes():
    table = pa.table({"text": ["  Hello\tWORLD ", None], "label": [1, 2]})
    a, b = run.transform(table, "text", ["label"], 10), run.transform(table, "text", ["label"], 10)
    assert a.equals(b)
    assert a.column("row_id").to_pylist() == [10, 11]
    assert a.column("text_normalized").to_pylist() == ["hello world", ""]
    assert a.column("n_words").to_pylist() == [2, 0]


def test_source_reads_across_row_groups(source_file):
    src = CountingSource(source_file)
    assert src.total == ROWS
    assert src.read(45, 105, ["label"]).column("label").to_pylist() == list(range(45, 105))
    capped = run.ParquetSource(["x"], lambda _: open(source_file, "rb"), max_rows=100)
    assert capped.total == 100


def test_full_shard_build_writes_parts_state_and_provenance(source_file, tmp_path):
    cfg, out = make_cfg(), tmp_path / "out"
    state = run.build_shard(cfg, "demo", "h", 1, CountingSource(source_file), run.LocalSink(out), resume=False,
                            card=("H\n{{VIEWER_CONFIGS}}\n{{CONFIG_SECTIONS}}", "{{REPOSITORY}} {{RUN_ID}} {{SOURCE_LICENSE}}"),
                            env={})
    start, stop = run.shard_range(ROWS, 3, 1)
    assert state["done"] and state["next_row"] == stop
    rows = [pq.read_table(out / p["path"]).column("row_id").to_pylist() for p in state["parts"]]
    assert sum(rows, []) == list(range(start, stop))
    assert json.loads((out / "_state/demo/shard-00001.json").read_text()) == state
    assert json.loads((out / "_provenance/demo/shard-00001.json").read_text())["run_id"] == "local"
    assert (out / "README.md").read_text() == ('H\n  - config_name: demo\n    data_files: "data/demo/**/*.parquet"\n'
                                              "<!-- gha-config: demo -->\nlocal local mit\n<!-- /gha-config: demo -->\n")
    assert state["code_sha"] == "local"


def test_resume_after_failure_skips_completed_parts_and_output_is_identical(source_file, tmp_path):
    cfg = make_cfg()
    ref = tmp_path / "ref"
    run.build_shard(cfg, "demo", "h", 0, CountingSource(source_file), run.LocalSink(ref), resume=True, env={})

    out = tmp_path / "out"
    with pytest.raises(RuntimeError, match="injected"):
        run.build_shard(cfg, "demo", "h", 0, CountingSource(source_file), FailingSink(out, ok=2), resume=True,
                        env={})
    partial = json.loads((out / "_state/demo/shard-00000.json").read_text())
    assert len(partial["parts"]) == 2 and not partial["done"]

    src = CountingSource(source_file)
    state = run.build_shard(cfg, "demo", "h", 0, src, run.LocalSink(out), resume=True, env={})
    assert state["done"]
    assert src.reads[0][0] == partial["next_row"]  # completed parts were not read again
    assert tree(out) == tree(ref)

    again = CountingSource(source_file)
    run.build_shard(cfg, "demo", "h", 0, again, FailingSink(out, ok=0), resume=True, env={})
    assert again.reads == []  # finished shard: no reads and no commit


def test_existing_state_needs_resume_and_matching_config(source_file, tmp_path):
    cfg, out = make_cfg(), tmp_path / "out"
    with pytest.raises(RuntimeError):
        run.build_shard(cfg, "demo", "h", 0, CountingSource(source_file), FailingSink(out, ok=1), resume=True,
                        env={})
    with pytest.raises(RuntimeError, match="--resume"):
        run.build_shard(cfg, "demo", "h", 0, CountingSource(source_file), run.LocalSink(out), resume=False, env={})
    with pytest.raises(RuntimeError, match="changed"):
        run.build_shard(cfg, "demo", "other", 0, CountingSource(source_file), run.LocalSink(out), resume=True,
                        env={})


def test_time_based_checkpoint_commits_partial_parts(source_file, tmp_path, monkeypatch):
    ticks = iter(range(0, 10**7, run.CHECKPOINT_SECONDS // 2))  # every clock() call advances 10 minutes
    monkeypatch.setattr(run, "READ_BATCH_ROWS", 30)
    cfg = make_cfg(checkpoint_rows=1000, num_shards=1)
    state = run.build_shard(cfg, "demo", "h", 0, CountingSource(source_file), run.LocalSink(tmp_path),
                            resume=False, env={}, clock=lambda: next(ticks))
    assert len(state["parts"]) > 1 and state["next_row"] == ROWS
    bounds = [p["rows"] for p in state["parts"]]
    assert bounds[0][0] == 0 and bounds[-1][1] == ROWS and all(a[1] == b[0] for a, b in zip(bounds, bounds[1:]))
    assert all(0 < hi - lo < cfg["checkpoint_rows"] for lo, hi in bounds)  # the clock, not the row count, cut them
    for part in state["parts"]:
        ids = pq.read_table(tmp_path / part["path"]).column("row_id").to_pylist()
        assert ids == list(range(*part["rows"]))


def test_retries_only_transient_errors():
    class HTTPErr(Exception):
        def __init__(self, status):
            self.response = type("R", (), {"status_code": status})()

    calls, sleeps = [], []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise HTTPErr(412)
        return "ok"
    assert run.with_retries(flaky, "x", sleep=sleeps.append) == "ok" and len(sleeps) == 2
    with pytest.raises(HTTPErr):
        run.with_retries(lambda: (_ for _ in ()).throw(HTTPErr(403)), "x", sleep=sleeps.append)
    assert len(sleeps) == 2
    assert run.is_retryable(ConnectionError()) and not run.is_retryable(ValueError())


def test_config_validation_and_card_placeholders():
    with pytest.raises(ValueError, match="revision"):
        make_cfg(source={"repo_id": "o/n", "revision": "main", "files": ["a.parquet"], "license": "mit"})
    with pytest.raises(ValueError, match="license"):
        make_cfg(source={"repo_id": "o/n", "revision": "a" * 40, "files": ["a.parquet"]})
    with pytest.raises(ValueError, match="placeholders"):
        run.render_section("{{NOPE}}", "demo", make_cfg(), 1, {})
    with pytest.raises(ValueError, match="placeholders"):
        run.merge_card("{{NOPE}} {{CONFIG_SECTIONS}}", None, "demo", "s", True)


def card_for(slug, cfg, old, env, replace=True):
    section = run.render_section(run.SECTION_TEMPLATE.read_text(), slug, cfg, 3000, env)
    return run.merge_card(run.CARD_TEMPLATE.read_text(), old, slug, section, replace)


def test_shipped_example_config_and_card_template_are_valid():
    slug, _, cfg = run.load_config(ROOT / "configs" / "example.yaml")
    assert slug == "example" and cfg["num_shards"] >= 1
    card = card_for(slug, cfg, None, {})
    assert "{{" not in card
    front = yaml.safe_load(card.split("---\n")[1])
    assert front["configs"] == [{"config_name": "example", "data_files": "data/example/**/*.parquet"}]


def test_card_keeps_every_configs_section_and_viewer_subset():
    cfg_a = make_cfg()
    cfg_b = make_cfg(source={"repo_id": "other/src", "revision": "b" * 40, "files": ["x.parquet"],
                             "license": "cc-by-4.0"})
    card = card_for("alpha", cfg_a, None, {"GITHUB_RUN_ID": "1"})
    card = card_for("beta", cfg_b, card, {"GITHUB_RUN_ID": "2"})
    assert "owner/name" in card and "other/src" in card and "cc-by-4.0" in card and "`mit`" in card
    front = yaml.safe_load(card.split("---\n")[1])
    assert [c["config_name"] for c in front["configs"]] == ["alpha", "beta"]
    assert front["configs"][1]["data_files"] == "data/beta/**/*.parquet"
    assert card_for("beta", cfg_b, card, {"GITHUB_RUN_ID": "2"}) is None  # unchanged: no commit
    assert card_for("alpha", cfg_a, card, {"GITHUB_RUN_ID": "9"}, replace=False) is None  # exists: keep it
    updated = card_for("alpha", cfg_a, card, {"GITHUB_RUN_ID": "9"})
    assert "/runs/9" in updated and "/runs/1)" not in updated and "other/src" in updated
    # the hf-init placeholder card (no sections) is replaced by the header plus the first section
    assert "gha-config: alpha" in card_for("alpha", cfg_a, "---\nlicense: mit\n---\nplaceholder\n", {})


def test_resume_refused_when_code_commit_changed(source_file, tmp_path):
    cfg, out = make_cfg(), tmp_path / "out"
    with pytest.raises(RuntimeError, match="injected"):
        run.build_shard(cfg, "demo", "h", 0, CountingSource(source_file), FailingSink(out, ok=1), resume=True,
                        env={"GITHUB_SHA": "a" * 40})
    with pytest.raises(RuntimeError, match="must not mix code versions"):
        run.build_shard(cfg, "demo", "h", 0, CountingSource(source_file), run.LocalSink(out), resume=True,
                        env={"GITHUB_SHA": "b" * 40})
    state = run.build_shard(cfg, "demo", "h", 0, CountingSource(source_file), run.LocalSink(out), resume=True,
                            env={"GITHUB_SHA": "a" * 40})
    assert state["done"] and state["code_sha"] == "a" * 40


def http_error(cls, status):
    request = httpx2.Request("GET", "https://huggingface.co/x")
    return cls(f"HTTP {status}", response=httpx2.Response(status, request=request))


class StubHfApi:
    """HfApi stand-in: hf_hub_download raises the queued errors, then serves files from a dict."""

    def __init__(self, tmp_path, files=None, errors=(), commit_errors=()):
        self.tmp, self.files, self.errors, self.commits = tmp_path, dict(files or {}), list(errors), []
        self.commit_errors, self.sha = list(commit_errors), "0" * 40

    def hf_hub_download(self, repo_id, filename, repo_type, revision=None):
        if self.errors:
            raise self.errors.pop(0)
        if filename not in self.files:
            raise http_error(RemoteEntryNotFoundError, 404)
        path = self.tmp / filename.replace("/", "_")
        path.write_text(self.files[filename])
        return str(path)

    def dataset_info(self, repo_id):
        return type("Info", (), {"sha": self.sha})()

    def create_commit(self, repo_id, operations, commit_message, repo_type, parent_commit=None):
        if self.commit_errors:
            raise self.commit_errors.pop(0)
        self.commits.append((parent_commit, {op.path_in_repo: op.path_or_fileobj for op in operations}))


@pytest.fixture
def no_sleep(monkeypatch):
    sleeps = []
    monkeypatch.setattr(run.time, "sleep", sleeps.append)
    return sleeps


def test_hub_state_404_means_no_state_but_outages_never_do(tmp_path, no_sleep):
    assert run.HubSink("o/d", StubHfApi(tmp_path)).read_json("_state/x.json") is None  # real 404
    flaky = StubHfApi(tmp_path, {"_state/x.json": '{"done": true}'},
                      errors=[LocalEntryNotFoundError("hub unreachable"), http_error(RemoteEntryNotFoundError, 503)])
    assert run.HubSink("o/d", flaky).read_json("_state/x.json") == {"done": True}  # retried, then read
    assert len(no_sleep) == 2
    down = StubHfApi(tmp_path, errors=[LocalEntryNotFoundError("hub unreachable")] * 6)
    with pytest.raises(LocalEntryNotFoundError):  # fails the job instead of restarting the shard from row 0
        run.HubSink("o/d", down).read_json("_state/x.json")


def test_hub_card_update_uses_parent_commit_and_retries_conflicts(tmp_path, no_sleep):
    api = StubHfApi(tmp_path, {"README.md": "old"}, commit_errors=[http_error(HfHubHTTPError, 412)])
    seen = []
    assert run.HubSink("o/d", api).update_card(lambda old: seen.append(old) or old + "+new") is True
    assert seen == ["old", "old"] and api.commits == [("0" * 40, {"README.md": b"old+new"})]
    assert run.HubSink("o/d", api).update_card(lambda old: None) is False and len(api.commits) == 1


def test_source_metadata_reads_are_retried(source_file, no_sleep):
    calls = []

    def opener(path):
        calls.append(path)
        if len(calls) == 1:
            raise ConnectionError("blip")
        return open(source_file, "rb")
    assert run.ParquetSource(["train.parquet"], opener).total == ROWS and len(no_sleep) == 1


def test_workflow_inputs_validation(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "ok.yaml").write_text("x: 1\n")
    inputs.check_config("ok", tmp_path)
    for bad in ("Bad", "../ok", "ok\n", "", "ok; rm -rf /"):
        with pytest.raises(ValueError):
            inputs.check_config(bad, tmp_path)
    assert inputs.parse_shards("[2,0,1]") == [0, 1, 2]
    for bad in ("[]", "[1,1]", "[true]", "[-1]", "[100000]", "0-2", json.dumps(list(range(257)))):
        with pytest.raises(ValueError):
            inputs.parse_shards(bad)
    assert inputs.parse_shard("7") == 7
    with pytest.raises(ValueError):
        inputs.parse_shard("7 ")


def test_inputs_main_writes_github_output(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    out = tmp_path / "out.txt"
    assert inputs.main(["dispatch"], {"CONFIG": "example", "SHARDS": "[1,0]", "GITHUB_OUTPUT": str(out)}) == 0
    assert out.read_text() == "shards=[0,1]\n"
    assert inputs.main(["dispatch"], {"CONFIG": "missing", "SHARDS": "[0]", "GITHUB_OUTPUT": str(out)}) == 1
    assert inputs.main(["shard"], {"CONFIG": "example", "SHARD": "x"}) == 1


def test_hub_log_identity_redacts_tokens_and_is_non_fatal(capsys):
    class WhoamiApi:
        def whoami(self):
            return {"name": "[OIDC]", "type": "user", "auth": {"type": "access_token", "accessToken": {
                "role": "fineGrained", "displayName": "hf_" + "a" * 34,
                "fineGrained": {"scoped": [{"entity": {"name": "o/d", "type": "dataset"},
                                            "permissions": ["repo.content.write"]}], "global": []}}}}

        def auth_check(self, repo_id, repo_type=None, write=False):
            if write:
                raise PermissionError("403 Forbidden")

    run.HubSink("o/d", WhoamiApi()).log_identity()
    out = capsys.readouterr().out
    assert '"name": "[OIDC]"' in out and "repo.content.write" in out and "hf_aaaa" not in out
    assert "read access to o/d: ok" in out and "write access to o/d: PermissionError" in out

    class BrokenApi:
        def whoami(self):
            raise RuntimeError("down")
    run.HubSink("o/d", BrokenApi()).log_identity()  # never raises
    assert "whoami failed (RuntimeError)" in capsys.readouterr().out
