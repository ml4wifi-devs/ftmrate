#!/usr/bin/scl enable devtoolset-8 rh-python38 -- /bin/bash -l

TOOLS_DIR="${TOOLS_DIR:=$HOME/ftmrate/tools}"

sbatch $TOOLS_DIR/slurm/run-classic-scenarios.sh
sbatch $TOOLS_DIR/slurm/run-ml-scenarios.sh
