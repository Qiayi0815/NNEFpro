#!/usr/bin/env bash
# Daily auto-commit (tracked files only) + push to GitHub.
# Uses "git add -u" so new/untracked files are never included unless you add them manually.

set -euo pipefail

# launchd 默认 PATH 不含 Homebrew；显式补上以免找不到 git
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"

REPO="/Library/Camille/FYP/nnef"
LOG="${HOME}/Library/Logs/nnef-git-daily-push.log"
export GIT_TERMINAL_PROMPT=0

mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1

ts() { date -u "+%Y-%m-%dT%H:%M:%SZ"; }

echo "=== $(ts) start ==="
cd "$REPO"

if ! git rev-parse --git-dir >/dev/null 2>&1; then
  echo "error: not a git repository: $REPO"
  exit 1
fi

branch="$(git branch --show-current)"
remote="$(git config "branch.${branch}.remote" || true)"
if [[ -z "$remote" ]]; then
  remote="origin"
fi

git add -u
if git diff --cached --quiet; then
  echo "no staged changes (tracked-only); skipping commit"
else
  git commit -m "chore: auto-sync $(ts)"
fi

git push "$remote" "$branch"
echo "=== $(ts) done ==="
