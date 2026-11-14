# gha-compute-smoke

This repository holds all the code that builds the public Hugging Face dataset
[sulabhkatiyar/gha-compute-smoke](https://huggingface.co/datasets/sulabhkatiyar/gha-compute-smoke). Builds run on GitHub-hosted
runners via [`.github/workflows/build.yml`](.github/workflows/build.yml); the dataset card links back
to the exact commit and workflow run that produced the data.

This is the **canonical reference example** of a gha-compute workload: a deliberately tiny,
end-to-end smoke test (3 shards of 1000 rows of `openai/gsm8k`, a trivial deterministic text transform)
that proves the setup works. New workloads are created from the same template and follow the
operator guidelines (`cloud_quota/github_actions.md` in the operator's utilities repository) and
[COMPLIANCE.md](COMPLIANCE.md).

## How a build works

1. The owner starts `build.yml` (`workflow_dispatch` only) with a committed config slug and a list of
   shard indices. There is no schedule, and workflows never trigger other workflows.
2. Inputs are re-validated, then each shard runs as one matrix job (at most 8 in parallel).
3. `builder/run.py` streams its slice of the pinned source dataset, applies a deterministic transform,
   and commits parquet parts with a resume state. A rerun skips parts that are already committed.
4. Uploads use a Hugging Face [Trusted Publisher](https://huggingface.co/docs/hub/trusted-publishers):
   the job's GitHub OIDC token is exchanged for a 60-minute token that can write only to
   `sulabhkatiyar/gha-compute-smoke`. The repository stores no secrets.

## Run it locally

```bash
python3.12 -m venv .venv && .venv/bin/pip install --require-hashes --no-deps -r requirements.lock
.venv/bin/pip install pytest
.venv/bin/python -m pytest tests
.venv/bin/python -m builder.run --config configs/example.yaml --shard 0 --local-dir out
```

## Licences

- Code: `MIT` (see [LICENSE](LICENSE)).
- Output data: `mit`. Each config pins its source dataset and records the source licence,
  which must allow redistribution.

See [COMPLIANCE.md](COMPLIANCE.md) for the rules this repository follows.
