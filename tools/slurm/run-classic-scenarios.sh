#!/usr/bin/scl enable devtoolset-8 rh-python38 -- /bin/bash -l

TOOLS_DIR="${TOOLS_DIR:=$HOME/ftmrate/tools}"

MANAGERS=("ns3::MinstrelHtWifiManager" "ns3::ThompsonSamplingWifiManager" "ns3::OracleWifiManager")
MANAGERS_NAMES=("Minstrel" "TS" "Oracle")
MANAGERS_LEN=${#MANAGERS[@]}

SEED_SHIFT=100

### Basic scenarios

run_equal_distance() {
  N_POINTS=9
  DISTANCE=$1

  for (( i = 0; i < MANAGERS_LEN; i++ )); do
    MANAGER=${MANAGERS[$i]}
    MANAGER_NAME=${MANAGERS_NAMES[$i]}
    ARRAY_SHIFT=0

    for (( j = 0; j < N_POINTS; j++)); do
      N_WIFI=$(( j == 0 ? 1 : 2 * j))
      SIM_TIME=$(( 10 * N_WIFI + 50 ))
      N_REP=$(( N_WIFI <= 4 ? 6 : N_WIFI * N_WIFI / 2 ))

      START=$ARRAY_SHIFT
      END=$(( ARRAY_SHIFT + N_REP - 1 ))

      ARRAY_SHIFT=$(( ARRAY_SHIFT + N_REP ))

      sbatch -p gpu --array=$START-$END "$TOOLS_DIR/slurm/distance/classic.sh" "$SEED_SHIFT" "$MANAGER" "$MANAGER_NAME" "$N_WIFI" "$DISTANCE" "$SIM_TIME"
    done
  done
}

run_hidden_node_distance() {
  N_REP=10
  N_POINTS=17
  N_WIFI=$1

  for (( i = 0; i < MANAGERS_LEN; i++ )); do
    MANAGER=${MANAGERS[$i]}
    MANAGER_NAME=${MANAGERS_NAMES[$i]}
    ARRAY_SHIFT=0

    for (( j = 1; j <= N_POINTS; j++)); do
      DISTANCE=$(( 5 * j + 20 ))
      SIM_TIME=$(( 20 * N_WIFI + 50 ))

      START=$ARRAY_SHIFT
      END=$(( ARRAY_SHIFT + N_REP - 1 ))

      ARRAY_SHIFT=$(( ARRAY_SHIFT + N_REP ))

      sbatch -p gpu --array=$START-$END "$TOOLS_DIR/slurm/hidden_node/classic.sh" "$SEED_SHIFT" "$MANAGER" "$MANAGER_NAME" "$N_WIFI" "$DISTANCE" "$SIM_TIME"
    done
  done
}

run_hidden_node_nwifi() {
  N_REP=30
  N_POINTS=10
  DISTANCE=$1
  RTS_CTS=$2

  for (( i = 0; i < MANAGERS_LEN; i++ )); do
    MANAGER=${MANAGERS[$i]}
    MANAGER_NAME=${MANAGERS_NAMES[$i]}
    ARRAY_SHIFT=0

    for (( j = 1; j <= N_POINTS; j++)); do
      N_WIFI=$(( j ))
      SIM_TIME=$(( 20 * N_WIFI + 50 ))

      START=$ARRAY_SHIFT
      END=$(( ARRAY_SHIFT + N_REP - 1 ))

      ARRAY_SHIFT=$(( ARRAY_SHIFT + N_REP ))

      sbatch -p gpu --array=$START-$END "$TOOLS_DIR/slurm/hidden_node/classic.sh" "$SEED_SHIFT" "$MANAGER" "$MANAGER_NAME" "$N_WIFI" "$DISTANCE" "$SIM_TIME" "$RTS_CTS"
    done
  done
}

run_rwpm() {
  N_REP=40
  N_WIFI=10
  SIM_TIME=1000
  NODE_SPEED=$1

  START=0
  END=$(( N_REP - 1 ))

  for (( i = 0; i < MANAGERS_LEN; i++ )); do
    MANAGER=${MANAGERS[$i]}
    MANAGER_NAME=${MANAGERS_NAMES[$i]}

    sbatch -p gpu --array=$START-$END "$TOOLS_DIR/slurm/rwpm/classic.sh" "$SEED_SHIFT" "$MANAGER" "$MANAGER_NAME" "$N_WIFI" "$SIM_TIME" "$NODE_SPEED"
  done

}

run_moving() {
  N_REP=15
  VELOCITY=$1
  SIM_TIME=$2
  INTERVAL=$3

  START=0
  END=$(( N_REP - 1 ))

  for (( i = 0; i < MANAGERS_LEN; i++ )); do
    MANAGER=${MANAGERS[$i]}
    MANAGER_NAME=${MANAGERS_NAMES[$i]}

    sbatch -p gpu --array=$START-$END "$TOOLS_DIR/slurm/moving/classic.sh" "$SEED_SHIFT" "$MANAGER" "$MANAGER_NAME" "$VELOCITY" "$SIM_TIME" "$INTERVAL"
  done
}

### var-power scenarios

run_power_static() {
  N_WIFI=$1
  DELTA=$2
  N_REP=10
  DISTANCE=7

  INTERVALS=("0.00215" "0.00464" "0.01" "0.02154" "0.04642" "0.1" "0.21544" "0.46416" "1.0" "2.15443" "4.64159" "10.0")
  SIM_TIMES=("20.0" "20.0" "20.1" "20.3" "20.5" "21.0" "22.2" "24.7" "30" "41.6" "66.5" "120")
  N_POINTS=${#INTERVALS[@]}   # mean_interval = a * q^n, a=0.001, q=2.15443469, n=1..12

  for (( i = 0; i < MANAGERS_LEN; i++ )); do
    MANAGER=${MANAGERS[$i]}
    MANAGER_NAME=${MANAGERS_NAMES[$i]}
    ARRAY_SHIFT=0

    for (( j = 0; j < N_POINTS; j++)); do
      INTERVAL=${INTERVALS[$j]}
      SIM_TIME=${SIM_TIMES[$j]}

      START=$ARRAY_SHIFT
      END=$(( ARRAY_SHIFT + N_REP - 1 ))

      ARRAY_SHIFT=$(( ARRAY_SHIFT + N_REP ))

      sbatch -p gpu --array=$START-$END "$TOOLS_DIR/slurm/power_static/classic.sh" "$SEED_SHIFT" "$MANAGER" "$MANAGER_NAME" "$N_WIFI" "$DISTANCE" "$DELTA" "$INTERVAL" "$SIM_TIME"
    done
  done
}

run_power_moving() {
  N_REP=15

  DELTA=$1
  INTERVAL=$2
  VELOCITY=$3
  START_POS=$4

  START=0
  END=$(( N_REP - 1 ))

  for (( i = 0; i < MANAGERS_LEN; i++ )); do
    MANAGER=${MANAGERS[$i]}
    MANAGER_NAME=${MANAGERS_NAMES[$i]}

    sbatch -p gpu --array=$START-$END "$TOOLS_DIR/slurm/power_moving/classic.sh" "$SEED_SHIFT" "$MANAGER" "$MANAGER_NAME" "$DELTA" "$INTERVAL" "$VELOCITY" "$START_POS"
  done
}

run_hybrid_moving() {
  N_REP=15

  VELOCITY="0.5"
  SIM_TIME=51
  INTERVAL="0.5"

  WALL_INTERVAL=5
  WALL_LOSS=3

  START=0
  END=$(( N_REP - 1 ))

  MANAGER="ns3::ThompsonSamplingWifiManager"
  MANAGER_NAME="TS"

  sbatch -p gpu --array=$START-$END "$TOOLS_DIR/slurm/moving/classic.sh" "$SEED_SHIFT" "$MANAGER" "$MANAGER_NAME" "$VELOCITY" "$SIM_TIME" "$INTERVAL" "$WALL_INTERVAL" "$WALL_LOSS"
}

### Run section

echo -e "\nQueue mobile stations scenario"
run_rwpm "1.4"                    # Fig. 12

echo -e "\nQueue equal distance (d=1) scenario"
run_equal_distance 1              # Fig. 13

echo -e "\nQueue hybrid moving station (v=0.5) scenario with walls"
run_hybrid_moving

## Legacy scenarios

# echo -e "\nQueue equal distance (d=20) scenario"
# run_equal_distance 20

# echo -e "\nQueue hidden node scenario with varying nWifi (RTS/CTS disabled)"
# run_hidden_node_nwifi 40 "false"

# echo -e "\nQueue hidden node scenario with varying nWifi (RTS/CTS enabled)"
# run_hidden_node_nwifi 40 "true"

# echo -e "\nQueue static stations scenario"
# run_rwpm 0

# echo -e "\nQueue moving station (v=1) scenario"
# run_moving 1 56 1

# echo -e "\nQueue moving station (v=2) scenario"
# run_moving 2 28 "0.5"

# echo -e "\nQueue hidden node scenario with varying distance"
# run_hidden_node_distance 1

# echo -e "\nQueue power with single static station (nWiFi=1, delta=15) scenario"
# run_power_static 1 15

# echo -e "\nQueue power with multiple static stations (nWiFi=10, delta=15) scenario"
# run_power_static 10 15

# echo -e "\nQueue power with moving station (delta=5, interval=4, v=0, start=5) scenario"
# run_power_moving 5 4 0 5

# echo -e "\nQueue power with moving station (delta=15, interval=8, v=0, start=5) scenario"
# run_power_moving 15 8 0 5
