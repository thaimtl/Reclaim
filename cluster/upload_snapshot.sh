#!/usr/bin/env bash
# Upload the code the cluster runs: a snapshot of this checkout to the bucket's repo/ (the cluster has no GitHub access;
# cluster/setup.sh pulls it from there). Run from a clean checkout of a pushed commit:
#   bash cluster/upload_snapshot.sh            refuses if there are uncommitted changes or HEAD is on no remote branch
#   FORCE=1 bash cluster/upload_snapshot.sh    uploads anyway, stamped -dirty / -unpushed (avoid: the run can't be traced)
#   DRY_RUN=1 bash cluster/upload_snapshot.sh  checks and prints the sync plan's summary, uploads nothing
# The snapshot holds the tracked files as they are in the working tree, plus cluster/VERSION = git rev-parse HEAD
# (+ -dirty, -unpushed), which setup.sh prints, so every cluster run can be traced back to a commit.
# Sync only adds and updates; nothing in the bucket is deleted.
set -euo pipefail
cd "$(dirname "$0")/.."
B=${RECLAIM_BUCKET:-hf://buckets/austinkomachi/Austin_CH}
HEAD=$(git rev-parse HEAD)
STAMP=$HEAD
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  [ "${FORCE:-0}" = "1" ] || { echo "uncommitted changes: commit and push first (or FORCE=1)"; git status --short; exit 1; }
  STAMP="$STAMP-dirty"
fi
git fetch -q origin || true
if [ -z "$(git branch -r --contains HEAD)" ]; then
  [ "${FORCE:-0}" = "1" ] || { echo "HEAD $HEAD is on no remote branch: push it first (or FORCE=1)"; exit 1; }
  STAMP="$STAMP-unpushed"
fi
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
git ls-files -z | tar --null -T - -cf - | tar -xf - -C "$STAGE"
echo "$STAMP" > "$STAGE/cluster/VERSION"
if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "dry run: $(git ls-files | wc -l) files as $STAMP"; hf sync "$STAGE" "$B/repo" --dry-run | head -1; exit 0
fi
echo "uploading $(git ls-files | wc -l) files as $STAMP to $B/repo"
hf sync "$STAGE" "$B/repo"
echo "done: the cluster's cluster/setup.sh will print code: $STAMP"
