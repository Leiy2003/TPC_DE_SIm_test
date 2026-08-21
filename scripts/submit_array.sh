#!/bin/bash
# Generic batch submitter that replaces sim_script/sim.sh, wf_gen*.sh and
# acceptance_waveform_gen.sh.
#
# Usage:
#   scripts/submit_array.sh <subcommand> <config> [--n-electrons N] [--start S] [--n-batches B] [--workers W]
#
# Subcommand is one of: de, cevns, acceptance, waveform.
#
# Examples:
#   scripts/submit_array.sh de configs/de_sim.yaml --start 0 --n-batches 200 --workers 50
#   scripts/submit_array.sh waveform configs/cevns_sim.yaml --n-electrons 5 --start 0 --n-batches 100
#
# Each batch consumes --files-per-batch=10 muon_track files (override with $FILES_PER_BATCH).

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <subcommand> <config> [extra args]" >&2
    exit 2
fi

SUBCMD="$1"; shift
CONFIG="$1"; shift

START=${START:-0}
N_BATCHES=${N_BATCHES:-1}
WORKERS=${WORKERS:-1}
FILES_PER_BATCH=${FILES_PER_BATCH:-10}
EXTRA_ARGS=("$@")

case "$SUBCMD" in
    de)         SCRIPT="scripts/run_de_sim.py" ;;
    cevns)      SCRIPT="scripts/run_cevns_sim.py" ;;
    acceptance) SCRIPT="scripts/run_acceptance.py" ;;
    waveform)   SCRIPT="scripts/run_waveform_gen.py" ;;
    *)
        echo "Unknown subcommand: $SUBCMD" >&2
        exit 2 ;;
esac

# Override START/N_BATCHES/WORKERS via long flags too.
while [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; do
    case "${EXTRA_ARGS[0]}" in
        --start)        START="${EXTRA_ARGS[1]}";        EXTRA_ARGS=("${EXTRA_ARGS[@]:2}") ;;
        --n-batches)    N_BATCHES="${EXTRA_ARGS[1]}";    EXTRA_ARGS=("${EXTRA_ARGS[@]:2}") ;;
        --workers)      WORKERS="${EXTRA_ARGS[1]}";      EXTRA_ARGS=("${EXTRA_ARGS[@]:2}") ;;
        *)
            break ;;
    esac
done

echo "Submitting ${SUBCMD} batches [${START}, $((START + N_BATCHES))) with ${WORKERS} parallel workers."

active=0
for ((batch=START; batch < START + N_BATCHES; batch++)); do
    python "${SCRIPT}" --config "${CONFIG}" --batch-index "${batch}" \
        --files-per-batch "${FILES_PER_BATCH}" "${EXTRA_ARGS[@]}" &
    ((active++)) || true
    if (( active >= WORKERS )); then
        wait -n
        ((active--)) || true
    fi
    sleep 1
done
wait

echo "All batches finished."
