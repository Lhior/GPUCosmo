#!/usr/bin/env bash
# Which parts of CI a pull request or push needs (testing-and-coverage.yml, job "changes").
# Writes code=true|false and docs=true|false to $GITHUB_OUTPUT:
#   docs  the change touches documentation (docs/, *.md, *.rst, COPYING*): build the docs
#   code  the change touches anything else: build the docs and run the tests
# Unknown paths count as code, and everything runs when the changed files cannot be determined
# (schedule, manual runs, a new branch, a failed diff): a wrong pattern only runs too much.
#
# Environment: EVENT (github.event_name), BASE and HEAD (the commits to compare).
set -uo pipefail

out="${GITHUB_OUTPUT:-/dev/stdout}"

everything() {
  echo "$1: everything runs"
  { echo "code=true"; echo "docs=true"; } >> "$out"
  exit 0
}

case "${EVENT:-}" in
  pull_request|push) ;;
  *) everything "event '${EVENT:-}'" ;;
esac
if [[ -z "${BASE:-}" || "${BASE}" =~ ^0+$ || -z "${HEAD:-}" ]]; then
  everything "no base commit"
fi
# three dots: the changes since the merge base (not the commits main gained meanwhile)
if ! files=$(git diff --name-only "${BASE}...${HEAD}"); then
  everything "git diff ${BASE}...${HEAD} failed"
fi
if [[ -z "$files" ]]; then
  everything "no changed files found"
fi

code=false
docs=false
echo "changed files:"
while IFS= read -r f; do
  echo "  $f"
  case "$f" in
    docs/*|*.md|*.rst|COPYING*) docs=true ;;
    *) code=true ;;
  esac
done <<< "$files"
echo "code=$code docs=$docs"
{ echo "code=$code"; echo "docs=$docs"; } >> "$out"
