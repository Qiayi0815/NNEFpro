# shellcheck shell=bash
# Resolve Yang decoy-eval checkpoint and --mixture_rama for evaluate_decoys.py.
#
# Inputs (optional env):
#   REPO          repo root (required)
#   YANG_CKPT     full path to model.pt — wins if set
#   YANG_RUN      run id dir: "yang_retrain_123", "runs/yang_retrain_123", or absolute
#   YANG_MIXTURE_RAMA  force 0 or 10 (paper release uses 0; train_yang_style retrains use 10)
#
# Sets:
#   YANG_CKPT
#   YANG_MIXTURE_RAMA
#
# Priority: YANG_CKPT > YANG_RUN > newest runs/yang_retrain_*/models/model.pt > params/exp1

_resolve_yang_checkpoint_main() {
  [[ -n "${REPO:-}" ]] || {
    echo '[resolve_yang_checkpoint] ERROR: REPO is not set' >&2
    return 1
  }

  if [[ -n "${YANG_CKPT:-}" ]]; then
    :
  elif [[ -n "${YANG_RUN:-}" ]]; then
    local _base
    case "$YANG_RUN" in
      /*) _base="$YANG_RUN" ;;
      runs/*) _base="$REPO/$YANG_RUN" ;;
      *) _base="$REPO/runs/$YANG_RUN" ;;
    esac
    YANG_CKPT="$_base/models/model.pt"
  else
    local _latest
    _latest="$(ls -dt "$REPO"/runs/yang_retrain_*/models/model.pt 2>/dev/null | head -1 || true)"
    if [[ -n "$_latest" && -f "$_latest" ]]; then
      YANG_CKPT="$_latest"
    else
      YANG_CKPT="$REPO/params/exp1/models/model.pt"
    fi
  fi

  if [[ -z "${YANG_MIXTURE_RAMA:-}" ]]; then
    if [[ "$YANG_CKPT" == *'/params/exp1/'* ]]; then
      YANG_MIXTURE_RAMA=0
    else
      YANG_MIXTURE_RAMA=10
    fi
  fi

  export YANG_CKPT YANG_MIXTURE_RAMA
}

_resolve_yang_checkpoint_main
