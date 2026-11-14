# Compliance checklist

This repository belongs to `sulabh-research`, a machine account used only for automated jobs. It uses
GitHub-hosted runners under the [GitHub Terms of Service](https://docs.github.com/en/site-policy/github-terms/github-terms-of-service)
and the [GitHub Actions terms](https://docs.github.com/en/site-policy/github-terms/github-terms-for-additional-products-and-features#actions).
Check every item before each change and each build.

## The workload
- [ ] It produces exactly one thing: the public dataset `sulabhkatiyar/gha-compute-smoke`, and the README says so.
- [ ] All code that produces it is in this repository. No private code, no downloaded scripts, no
      "run an arbitrary command" inputs.
- [ ] Inputs are public datasets whose licences allow redistribution. Each config pins a source revision
      and records its licence, which appears in the dataset card.
- [ ] Output data licence (`mit`) is compatible with every source licence.
- [ ] Nothing here mines, scans, attacks, serves or proxies traffic, trains or evaluates unrelated models,
      or uses Actions as storage or a CDN.

## The workflow
`gha.py` lints every file in `.github/workflows/` before `init-repo`, `git push` and `dispatch`, and
refuses if the trigger, permissions, `uses:` pins or `max-parallel` items below do not hold.

- [ ] Trigger is `workflow_dispatch` only: no `schedule`, `push`, `pull_request*` or `workflow_run`, and
      no step dispatches workflows.
- [ ] Top-level `permissions: contents: read`; `id-token: write` only on the build job.
- [ ] Only official `actions/*` actions, pinned by full commit SHA; `persist-credentials: false`.
- [ ] Inputs reach scripts only through `env:` and are re-validated first; never `${{ inputs.* }}` in `run:`.
- [ ] Dependencies installed with `--require-hashes` from `requirements.lock`.
- [ ] `max-parallel: 8`, `fail-fast: false`, `timeout-minutes: 340`, `concurrency` per config, and
      the run-attempt guard (more than 4 attempts fails).

## The data
- [ ] The dataset is public, gated `auto`, and its card links the repository, commit SHA, workflow and run.
- [ ] Logs never print data rows or secrets.

## Never
- [ ] Delete, transfer or rename repositories from automation (a user action in the web UI only).
- [ ] Use this account or its token manually, or create extra accounts or organizations.
