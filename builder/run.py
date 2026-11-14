"""Build one shard: stream rows of a public HF dataset, apply a deterministic transform, write parquet parts.

  python -m builder.run --config configs/<slug>.yaml --shard K [--resume] [--local-dir DIR]

Writes to the HF dataset named by $HF_DATASET with the fine-grained token in $HF_TOKEN (the Actions
secret HF_TOKEN_GHA, which can write only to that dataset; source reads stay anonymous), or to DIR for a
local dry run:
  data/<slug>/shard-0000K/part-XXXX.parquet   one part per checkpoint
  _state/<slug>/shard-0000K.json              resume state, committed atomically with each part
  _provenance/<slug>/shard-0000K.json         repo, commit SHA, workflow, run id/attempt, source
  README.md                                   dataset card (DATASET_CARD*.md.tmpl): one section and one
                                              viewer subset per config, updated when a shard finishes
Logs carry counts only: never rows, never tokens.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import random
import re
import time
import unicodedata

import httpx2
from huggingface_hub.errors import LocalEntryNotFoundError, RemoteEntryNotFoundError
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

SLUG_RE = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")
CHECKPOINT_SECONDS = 20 * 60
READ_BATCH_ROWS = 1000
RETRY_STATUS = {408, 409, 412, 429, 500, 502, 503, 504}
CARD_TEMPLATE = Path(__file__).resolve().parent.parent / "DATASET_CARD.md.tmpl"
SECTION_TEMPLATE = Path(__file__).resolve().parent.parent / "DATASET_CARD_SECTION.md.tmpl"
SECTION_RE = re.compile(r"<!-- gha-config: ([a-z0-9][a-z0-9_-]{0,63}) -->\n.*?<!-- /gha-config: \1 -->\n", re.S)


def log(message):
    print(message, flush=True)


# ---------------------------------------------------------------- config

def validate_config(cfg):
    """Return cfg if it is a well-formed builder config, else raise ValueError."""
    src = cfg.get("source") if isinstance(cfg, dict) else None
    if not isinstance(src, dict):
        raise ValueError("config needs a source mapping")
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", str(src.get("repo_id", ""))):
        raise ValueError("source.repo_id must be <owner>/<name>")
    if not re.fullmatch(r"[0-9a-f]{40}", str(src.get("revision", ""))):
        raise ValueError("source.revision must be a pinned 40-hex commit")
    files = src.get("files")
    if not isinstance(files, list) or not files or not all(
            isinstance(f, str) and f.endswith(".parquet") and ".." not in f.split("/") for f in files):
        raise ValueError("source.files must be a non-empty list of relative .parquet paths")
    if not isinstance(src.get("license"), str) or not src["license"]:
        raise ValueError("source.license is required (inputs must be redistributable)")
    if not isinstance(cfg.get("text_column"), str):
        raise ValueError("text_column is required")
    keep = cfg.setdefault("keep_columns", [])
    if not isinstance(keep, list) or not all(isinstance(c, str) for c in keep):
        raise ValueError("keep_columns must be a list of column names")
    for key, low in (("num_shards", 1), ("checkpoint_rows", 1)):
        if type(cfg.get(key)) is not int or cfg[key] < low:
            raise ValueError(f"{key} must be an integer >= {low}")
    if cfg.get("max_rows") is not None and (type(cfg["max_rows"]) is not int or cfg["max_rows"] < 1):
        raise ValueError("max_rows must be a positive integer when set")
    return cfg


def load_config(path):
    path = Path(path)
    if not SLUG_RE.fullmatch(path.stem):
        raise ValueError(f"config file name must be <slug>.yaml with slug matching {SLUG_RE.pattern}")
    raw = path.read_bytes()
    return path.stem, hashlib.sha256(raw).hexdigest(), validate_config(yaml.safe_load(raw))


def shard_range(total, num_shards, shard):
    """Contiguous [start, stop) of global rows for shard; shards tile 0..total without overlap."""
    if not 0 <= shard < num_shards:
        raise ValueError(f"shard {shard} is outside 0..{num_shards - 1}")
    start, stop = total * shard // num_shards, total * (shard + 1) // num_shards
    if start == stop:
        raise ValueError(f"shard {shard} is empty: num_shards {num_shards} > rows {total}")
    return start, stop


# ---------------------------------------------------------------- retries

def is_retryable(exc):
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status is not None:
        return status in RETRY_STATUS
    # hf_hub_download raises LocalEntryNotFoundError when the Hub is unreachable or answers 5xx: transient.
    return isinstance(exc, (ConnectionError, TimeoutError, httpx2.TransportError, LocalEntryNotFoundError))


def with_retries(fn, what, attempts=6, base_delay=2.0, sleep=None):
    """Call fn, retrying transient HTTP/network errors with exponential backoff and jitter."""
    sleep = sleep or time.sleep
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:
            if attempt == attempts or not is_retryable(exc):
                raise
            delay = base_delay * 2 ** (attempt - 1) * random.uniform(0.5, 1.5)
            status = getattr(getattr(exc, "response", None), "status_code", "-")
            log(f"{what}: {type(exc).__name__} (status {status}), attempt {attempt}/{attempts}; "
                f"retrying in {delay:.0f}s")
            sleep(delay)


# ---------------------------------------------------------------- source and transform

class ParquetSource:
    """Parquet files read as one global row sequence, one row group at a time (streamed, not downloaded)."""

    def __init__(self, paths, opener, max_rows=None):
        self.opener = opener
        self.groups = []  # (path, row_group, first_global_row, num_rows)
        offset = 0
        for path in paths:
            def footer(path=path):
                with opener(path) as fh:
                    return pq.ParquetFile(fh).metadata
            meta = with_retries(footer, f"read {path} metadata")
            for group in range(meta.num_row_groups):
                rows = meta.row_group(group).num_rows
                self.groups.append((path, group, offset, rows))
                offset += rows
        self.total = offset if max_rows is None else min(offset, max_rows)
        self._cached = (None, None)

    def _group(self, path, group, columns):
        key = (path, group, tuple(columns))
        if self._cached[0] != key:
            def read():
                with self.opener(path) as fh:
                    return pq.ParquetFile(fh).read_row_group(group, columns=columns)
            self._cached = (key, with_retries(read, f"read {path} row group {group}"))
        return self._cached[1]

    def read(self, start, stop, columns):
        pieces = []
        for path, group, first, rows in self.groups:
            if first < stop and first + rows > start:
                lo, hi = max(start - first, 0), min(stop - first, rows)
                pieces.append(self._group(path, group, columns).slice(lo, hi - lo))
        return pa.concat_tables(pieces)


def hf_source(cfg):
    """Anonymous (token=False) reads of the pinned source revision: the scoped write token is never sent."""
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem(token=False)
    src = cfg["source"]
    prefix = f"datasets/{src['repo_id']}@{src['revision']}/"
    return ParquetSource(src["files"], lambda path: fs.open(prefix + path, "rb"), cfg.get("max_rows"))


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", "" if text is None else str(text)).split()).casefold()


def transform(table, text_column, keep_columns, first_row_id):
    """Deterministic CPU transform: normalized text plus simple per-row statistics."""
    texts = [normalize(t) for t in table.column(text_column).to_pylist()]
    columns = {"row_id": pa.array(range(first_row_id, first_row_id + table.num_rows), pa.int64())}
    columns.update({name: table.column(name) for name in keep_columns})
    columns["text_normalized"] = pa.array(texts, pa.string())
    columns["n_chars"] = pa.array([len(t) for t in texts], pa.int32())
    columns["n_words"] = pa.array([len(t.split()) for t in texts], pa.int32())
    columns["text_sha256"] = pa.array([hashlib.sha256(t.encode()).hexdigest() for t in texts], pa.string())
    return pa.table(columns)


def parquet_bytes(table):
    buf = io.BytesIO()
    pq.write_table(table, buf, compression="zstd")
    return buf.getvalue()


# ---------------------------------------------------------------- sinks

class LocalSink:
    """Dry-run sink: files under a local directory, each written atomically, in the given order."""

    def __init__(self, root):
        self.root = Path(root)

    def read_json(self, path):
        target = self.root / path
        return json.loads(target.read_text(encoding="utf-8")) if target.exists() else None

    def commit(self, files, message):
        for path, data in files.items():
            target = self.root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name + ".tmp")
            tmp.write_bytes(data)
            os.replace(tmp, target)

    def update_card(self, update):
        """update(old README text or None) -> new text, or None for no change. Returns True if written."""
        target = self.root / "README.md"
        new = update(target.read_text(encoding="utf-8") if target.exists() else None)
        if new is not None:
            LocalSink.commit(self, {"README.md": new.encode()}, "card")
        return new is not None


class HubSink:
    """The HF dataset, written with an explicit token (state reads and commits alike)."""

    def __init__(self, repo_id, api=None, token=None):
        if api is None:
            from huggingface_hub import HfApi
            api = HfApi(token=token)
        self.api, self.repo_id = api, repo_id

    def _download_text(self, path, revision=None):
        """File text, or None only for a real 404 (RemoteEntryNotFoundError). A Hub outage or 5xx surfaces
        as LocalEntryNotFoundError, which is retried and then fails the job: never "no state"."""
        try:
            local = with_retries(lambda: self.api.hf_hub_download(self.repo_id, path, repo_type="dataset",
                                                                  revision=revision), f"download {path}")
        except RemoteEntryNotFoundError:
            return None
        return Path(local).read_text(encoding="utf-8")

    def log_identity(self):
        """Log who the uploads run as (user name, token role and scopes; never the token itself). Non-fatal."""
        try:
            info = self.api.whoami()
        except Exception as exc:  # diagnostics only
            log(f"upload identity: whoami failed ({type(exc).__name__})")
            return
        auth = info.get("auth") or {}
        tok = auth.get("accessToken") or {}
        scopes = [(s.get("entity", {}).get("name"), s.get("entity", {}).get("type"), s.get("permissions"))
                  for s in (tok.get("fineGrained") or {}).get("scoped") or []]
        summary = {"name": info.get("name"), "type": info.get("type"), "auth_type": auth.get("type"),
                   "role": tok.get("role"), "scoped": scopes,
                   "global": (tok.get("fineGrained") or {}).get("global")}
        log("upload identity: " + re.sub(r"hf_[A-Za-z0-9_]+", "hf_***", json.dumps(summary, default=str)))
        for write in (False, True):
            try:
                self.api.auth_check(self.repo_id, repo_type="dataset", write=write)
                result = "ok"
            except Exception as exc:  # diagnostics only
                result = type(exc).__name__
            log(f"upload identity: {'write' if write else 'read'} access to {self.repo_id}: {result}")

    def read_json(self, path):
        text = self._download_text(path)
        return None if text is None else json.loads(text)

    def commit(self, files, message):
        from huggingface_hub import CommitOperationAdd

        def attempt():  # fresh operations per attempt
            ops = [CommitOperationAdd(path_in_repo=p, path_or_fileobj=data) for p, data in files.items()]
            return self.api.create_commit(self.repo_id, ops, commit_message=message, repo_type="dataset")
        with_retries(attempt, f"commit {message!r}")

    def update_card(self, update):
        """Read-modify-write README.md under an optimistic lock (parent_commit): a concurrent commit makes
        the write fail with a retryable conflict and the card is re-read, so no config's section is lost."""
        from huggingface_hub import CommitOperationAdd

        def attempt():
            head = self.api.dataset_info(self.repo_id).sha
            new = update(self._download_text("README.md", revision=head))
            if new is None:
                return False
            self.api.create_commit(self.repo_id, [CommitOperationAdd(path_in_repo="README.md",
                                                                     path_or_fileobj=new.encode())],
                                   commit_message="Update dataset card", repo_type="dataset", parent_commit=head)
            return True
        return with_retries(attempt, "update README.md", attempts=8)


# ---------------------------------------------------------------- build

def provenance(slug, shard, cfg, env):
    return {
        "repository": env.get("GITHUB_REPOSITORY", "local"),
        "sha": env.get("GITHUB_SHA", "local"),
        "workflow_ref": env.get("GITHUB_WORKFLOW_REF", "local"),
        "run_id": env.get("GITHUB_RUN_ID", "local"),
        "run_attempt": env.get("GITHUB_RUN_ATTEMPT", "local"),
        "config": slug,
        "shard": shard,
        "source": {k: cfg["source"][k] for k in ("repo_id", "revision", "files", "license")},
    }


def render_section(template, slug, cfg, total, env):
    """This config's card section (between gha-config markers) from DATASET_CARD_SECTION.md.tmpl."""
    prov = provenance(slug, None, cfg, env)
    server = env.get("GITHUB_SERVER_URL", "https://github.com")
    values = {
        "REPOSITORY": prov["repository"], "SHA": prov["sha"], "WORKFLOW_REF": prov["workflow_ref"],
        "RUN_ID": prov["run_id"], "RUN_URL": f"{server}/{prov['repository']}/actions/runs/{prov['run_id']}",
        "CONFIG": slug, "NUM_SHARDS": str(cfg["num_shards"]), "ROWS": str(total),
        "SOURCE_REPO": cfg["source"]["repo_id"], "SOURCE_REVISION": cfg["source"]["revision"],
        "SOURCE_LICENSE": cfg["source"]["license"],
    }
    text = template
    for key, value in values.items():
        text = text.replace("{{" + key + "}}", value)
    if "{{" in text:
        raise ValueError(f"dataset card template has unknown placeholders: {re.findall(r'{{[^}]*}}', text)}")
    return f"<!-- gha-config: {slug} -->\n{text.strip()}\n<!-- /gha-config: {slug} -->\n"


def merge_card(header, old, slug, section, replace=True):
    """Card text with this config's section added (or replaced) and every other config's section kept; one
    viewer subset per config (data/<slug>/). None when nothing changes (or replace=False and it exists)."""
    sections = {m[1]: m[0] for m in SECTION_RE.finditer(old or "")}
    if slug in sections and not replace:
        return None
    sections[slug] = section
    if "{{" in header.replace("{{VIEWER_CONFIGS}}", "").replace("{{CONFIG_SECTIONS}}", ""):
        raise ValueError("dataset card header has unknown placeholders")
    viewer = "\n".join(f'  - config_name: {s}\n    data_files: "data/{s}/**/*.parquet"' for s in sorted(sections))
    text = header.replace("{{VIEWER_CONFIGS}}", viewer).replace(
        "{{CONFIG_SECTIONS}}", "\n".join(sections[s] for s in sorted(sections)))
    return None if text == old else text


def build_shard(cfg, slug, config_sha256, shard, source, sink, resume, card=None,
                env=os.environ, clock=time.monotonic):
    """Build (or resume) one shard; returns the final state. Each part is committed with its state.
    card = (header template, section template) updates README.md when the shard is done."""
    start, stop = shard_range(source.total, cfg["num_shards"], shard)
    code_sha = env.get("GITHUB_SHA", "local")
    name = f"shard-{shard:05d}"
    data_dir, state_path, prov_path = f"data/{slug}/{name}", f"_state/{slug}/{name}.json", \
        f"_provenance/{slug}/{name}.json"
    state = sink.read_json(state_path)
    if state is None:
        state = {"config": slug, "config_sha256": config_sha256, "code_sha": code_sha, "shard": shard,
                 "range": [start, stop], "next_row": start, "parts": [], "done": False}
    elif not resume:
        raise RuntimeError(f"{state_path} exists; pass --resume to continue it")
    elif state.get("config_sha256") != config_sha256 or state.get("range") != [start, stop]:
        raise RuntimeError(f"configs/{slug}.yaml changed since {name} started; use a new config slug")
    elif state.get("code_sha") != code_sha:
        raise RuntimeError(f"{name} was built at commit {state.get('code_sha')} but this run is at {code_sha}; "
                           f"one shard must not mix code versions: build under a new config slug "
                           f"(e.g. copy configs/{slug}.yaml to configs/{slug}-v2.yaml)")

    def update_card(replace):
        if card is not None:
            section = render_section(card[1], slug, cfg, source.total, env)
            if sink.update_card(lambda old: merge_card(card[0], old, slug, section, replace)):
                log(f"{name}: dataset card section for {slug} written")
    if state["done"]:
        log(f"{name}: all {len(state['parts'])} parts already complete; skipping (no data commit)")
        update_card(replace=False)  # only if the section is missing (e.g. an earlier card update failed)
        return state
    if state["parts"]:
        log(f"{name}: resuming at row {state['next_row']}; {len(state['parts'])} completed parts skipped")
    columns = list(dict.fromkeys([cfg["text_column"], *cfg["keep_columns"]]))
    batch = min(cfg["checkpoint_rows"], READ_BATCH_ROWS)
    row = state["next_row"]
    while row < stop:
        part_start, tables, began = row, [], clock()
        while row < stop and row - part_start < cfg["checkpoint_rows"] and clock() - began < CHECKPOINT_SECONDS:
            n = min(batch, stop - row, cfg["checkpoint_rows"] - (row - part_start))
            tables.append(transform(source.read(row, row + n, columns), cfg["text_column"],
                                    cfg["keep_columns"], row))
            row += n
        data = parquet_bytes(pa.concat_tables(tables))
        part_path = f"{data_dir}/part-{len(state['parts']):04d}.parquet"
        new_state = {**state, "next_row": row, "done": row >= stop, "parts": state["parts"] + [
            {"path": part_path, "rows": [part_start, row], "sha256": hashlib.sha256(data).hexdigest()}]}
        files = {part_path: data,
                 prov_path: json.dumps(provenance(slug, shard, cfg, env), indent=2).encode()}
        files[state_path] = json.dumps(new_state, indent=2).encode()  # last: a local sink writes it last
        sink.commit(files, f"{slug} {name}: part {len(state['parts'])} rows {part_start}-{row}")
        state = new_state
        log(f"{name}: committed part {len(state['parts']) - 1} ({row - part_start} rows; {row - start}/"
            f"{stop - start} done)")
    update_card(replace=True)
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--resume", action="store_true", help="continue from _state/ (skip completed parts)")
    parser.add_argument("--local-dir", help="write to this directory instead of the HF dataset (dry run)")
    args = parser.parse_args(argv)
    slug, config_sha256, cfg = load_config(args.config)
    if args.local_dir:
        sink = LocalSink(args.local_dir)
    elif os.environ.get("HF_DATASET"):
        token = os.environ.get("HF_TOKEN", "").strip()
        if not token:
            raise SystemExit("HF_TOKEN is empty: the repo needs the Actions secret HF_TOKEN_GHA (the owner runs "
                             "gha.py set-secret), passed as HF_TOKEN in the build step's env")
        sink = HubSink(os.environ["HF_DATASET"], token=token)
        sink.log_identity()
    else:
        parser.error("set HF_DATASET (workflow) or pass --local-dir")
    source = hf_source(cfg)
    log(f"config {slug}: {source.total} rows from {cfg['source']['repo_id']}@{cfg['source']['revision'][:12]}, "
        f"{cfg['num_shards']} shards; building shard {args.shard}")
    build_shard(cfg, slug, config_sha256, args.shard, source, sink, args.resume,
                card=(CARD_TEMPLATE.read_text(encoding="utf-8"), SECTION_TEMPLATE.read_text(encoding="utf-8")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
