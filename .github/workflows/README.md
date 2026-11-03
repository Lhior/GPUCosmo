# Workflows

CI runs on two kinds of machine, split by trust:

* **GitHub-hosted runners** (`ubuntu-latest`, CPU, no data products) run everything a pull request
  triggers: `changes`, `lint`, `docs` and the unit tests without data on Python 3.11 and 3.13.
  Tests that need data products, a GPU or git-only extras skip there.
* **One self-hosted runner**, `gpucosmo-laptop` (labels `self-hosted, gpu, cuda`; RTX 3070 Laptop,
  16 threads, ~40 GB of likelihood data), runs only trusted code: pushes to `main`, the nightly
  schedule and manual runs, which only collaborators with write access can start. Those jobs are
  the unit tests with data and the slow GPU tests. They are tied to the development repository
  (`github.repository == 'Lhior/GPUCosmo-dev'`), so in this repository only the GitHub-hosted jobs run.

## Security

GitHub's guidance (<https://docs.github.com/en/actions/reference/security/secure-use>, "Hardening
for self-hosted runners") is that self-hosted runners "should almost never be used for public
repositories", because anyone can open a pull request, and a self-hosted runner is not an
ephemeral clean machine. The workflows therefore follow these rules:

* No job on the self-hosted runner runs on `pull_request` (or `pull_request_target`). Each such
  job has `if: github.event_name != 'pull_request' && github.repository == 'Lhior/GPUCosmo-dev'` (the
  second condition keeps forks, and this public repository, from looking for the runner).
* To run the GPU and data jobs on a contribution, review it, push the reviewed commits to a
  branch of this repository and start a manual run, or merge it.
* The `GITHUB_TOKEN` defaults to `contents: read`. The self-hosted test jobs check out with
  `persist-credentials: false`.
* Event data reaches scripts through environment variables, never `${{ }}` inside `run:`.
* Third-party actions are pinned to full commit SHAs (the tag in a comment; Dependabot updates
  both). GitHub's own `actions/*` stay on major tags.
* `actionlint` (config: `.github/actionlint.yaml`) and the `check-github-workflows` pre-commit hook
  validate the files.

Repository settings: fork pull-request workflows from outside collaborators need approval; no
secrets are exposed to fork pull requests; branch protection on `main` requires the hosted
`lint`, `docs` and `unit (py*, no data)` checks.

**Starting a run.** Pull requests, pushes to `main` and the nightly schedule start CI. Start a run by hand with
`gh workflow run testing-and-coverage.yml --ref <branch>` (or *Run workflow* in the Actions tab),
then `gh run watch`; a manual run includes every job. (The workflow keeps the file name
`testing-and-coverage.yml` so it can be dispatched on a branch: GitHub dispatches only workflows
whose file exists on `main`, then runs the branch's version of it.)

**What runs.** The first job, `changes` (`.github/ci/changed_paths.sh`), sorts the files a pull
request or push changes: documentation (`docs/`, `*.md`, `*.rst`, `COPYING*`) or code (anything
else). A documentation-only change runs `lint` and `docs`; a code change runs the jobs
below. The nightly schedule and manual runs always run everything, and so does any case where the
changed files cannot be determined. Jobs are skipped with `if:` (skipped jobs count as passed), not
the whole workflow with `paths:`, so required checks would not stay pending.

| workflow | when | runner | what |
|---|---|---|---|
| `testing-and-coverage.yml` (CI) | PR, push to main, nightly 03:00 EDT, manual | GitHub-hosted | `lint` (pre-commit); `docs` (Sphinx) on documentation or code changes; `unit (py3.11/py3.13, no data)` (pytest `-m "not slow"`, CPU, clean install; data-dependent tests must skip) on code changes |
| `testing-and-coverage.yml` (CI) | push to main (code changes), nightly, manual — never a PR | self-hosted | `unit (py3.13, data)` (pytest `-m "not slow"`, CPU, with data), `slow tests (GPU)` (pytest `-m slow`, `JAX_PLATFORMS=cuda,cpu`) |
| `publish-to-pypi.yml` | manual only | GitHub-hosted | build and publish (trusted publishing); the preview is not on PyPI |

While the repository is private, the GitHub-hosted jobs use the account's Actions minutes; on a
public repository they are free.

The self-hosted jobs (tests with data, slow GPU tests) run only in the private development
repository `Lhior/GPUCosmo-dev`, on its own runner; `.github/actions/setup-gpucosmo` builds their
environment (a fresh uv venv, the package installed editable, and the machine's data linked into
`data/` and `packages/`).
