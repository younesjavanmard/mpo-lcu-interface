#!/usr/bin/env bash
# Run all round-2 tasks, 8 in parallel, single-threaded BLAS each.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
mkdir -p results logs
# Note: chi=32 sampling of C2H2/C2H4 at N=16000 takes ~2.5 h per seed; in the paper
# these points were run as separate per-seed processes (seeds 0-2, --seeds/--out).
cat tasks.txt | xargs -P 8 -L 1 bash -c 'python run_round2.py --mol "$0" --chi "$1" --nsamples "$2" > "logs/$0_chi$1.log" 2>&1; echo "finished $0 chi=$1 rc=$?" >> logs/queue.log'
echo ALLDONE >> logs/queue.log
