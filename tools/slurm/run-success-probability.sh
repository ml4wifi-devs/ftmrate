#!/usr/bin/scl enable devtoolset-11 rh-python38 -- /bin/bash -l

TOOLS_DIR="${TOOLS_DIR:=$HOME/ftmrate/tools}"

MIN_SNR=5
MAX_SNR=55

MIN_MODE=0
MAX_MODE=11

SEED=100

for (( snr = MIN_SNR; snr <= MAX_SNR; snr++ )); do
  for (( mode = MIN_MODE; mode <= MAX_MODE; mode++)); do
    sbatch -p gpu "$TOOLS_DIR/slurm/success-probability/classic.sh" "$SEED" "$mode" "$snr"
    SEED=$(( SEED + 1 ))
  done
done
