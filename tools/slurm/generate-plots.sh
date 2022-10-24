#!/usr/bin/scl enable devtoolset-11 rh-python38 -- /bin/bash -l

TOOLS_DIR="${TOOLS_DIR:=$HOME/tools}"
export PYTHONPATH="$PYTHONPATH:$TOOLS_DIR"

bash $TOOLS_DIR/outputs/combine_csv_files.sh
python3 $TOOLS_DIR/plots/moving_plots.py
python3 $TOOLS_DIR/plots/moving_t-tests.py
python3 $TOOLS_DIR/plots/rwpm_plots.py
python3 $TOOLS_DIR/plots/rwpm_t-tests.py
python3 $TOOLS_DIR/plots/static_plots.py
python3 $TOOLS_DIR/plots/static_t-tests.py

echo "Done!"
