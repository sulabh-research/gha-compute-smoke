"""Re-validate workflow_dispatch inputs. Standard library only: runs before dependencies are installed.

  python3 -m builder.inputs dispatch   # env CONFIG, SHARDS; writes shards=<json list> to $GITHUB_OUTPUT
  python3 -m builder.inputs shard      # env CONFIG, SHARD
"""
import json
import os
from pathlib import Path
import re
import sys

SLUG_RE = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")
MAX_SHARDS = 256
MAX_SHARD_INDEX = 99999


def check_config(slug, root="."):
    if not SLUG_RE.fullmatch(slug):
        raise ValueError(f"config slug must match {SLUG_RE.pattern}")
    if not (Path(root) / "configs" / f"{slug}.yaml").is_file():
        raise ValueError(f"configs/{slug}.yaml does not exist")


def parse_shard(text):
    if not re.fullmatch(r"\d{1,5}", text):
        raise ValueError("shard must be an integer in 0..99999")
    return int(text)


def parse_shards(text):
    try:
        shards = json.loads(text)
    except json.JSONDecodeError:
        raise ValueError("shards must be a JSON list of integers") from None
    if not isinstance(shards, list) or not shards or not all(type(s) is int for s in shards):
        raise ValueError("shards must be a non-empty JSON list of integers")
    if len(set(shards)) != len(shards) or len(shards) > MAX_SHARDS:
        raise ValueError(f"shards must be unique and at most {MAX_SHARDS}")
    if not all(0 <= s <= MAX_SHARD_INDEX for s in shards):
        raise ValueError(f"shard indices must be within 0..{MAX_SHARD_INDEX}")
    return sorted(shards)


def main(argv=None, env=None):
    argv = sys.argv[1:] if argv is None else argv
    env = os.environ if env is None else env
    try:
        if argv not in (["dispatch"], ["shard"]):
            raise ValueError("usage: python3 -m builder.inputs dispatch|shard")
        check_config(env.get("CONFIG", ""))
        if argv == ["shard"]:
            print(f"inputs ok: config={env['CONFIG']} shard={parse_shard(env.get('SHARD', ''))}")
            return 0
        shards = parse_shards(env.get("SHARDS", ""))
        with open(env["GITHUB_OUTPUT"], "a", encoding="utf-8") as out:
            out.write(f"shards={json.dumps(shards, separators=(',', ':'))}\n")
        print(f"inputs ok: config={env['CONFIG']} shards={len(shards)}")
        return 0
    except ValueError as err:
        print(f"::error::invalid input: {err}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
