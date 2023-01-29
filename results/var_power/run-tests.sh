#!/usr/bin/env bash

DISTANCES=(0)
DISTANCES_LEN=${#DISTANCES[@]}

POWERS=(10)
POWERS_LEN=${#POWERS[@]}

for (( d = 0; d < DISTANCES_LEN; d++ )); do
    for (( p = 0; p < POWERS_LEN; p++ )); do

        DIST=${DISTANCES[$d]}
        POW=${POWERS[$p]}

        # Constant power
        python3 ml4wifi/envs/ml_wifi_manager.py --scenario="moving" --ml_manager="kf" --managerName="KF" --velocity="0" --startPosition="$DIST" --simulationTime="40" --warmupTime="4" --fuzzTime="2" --measurementsInterval="0.5" --lossModel="Nakagami" --seed="11227" --interval="4" --delta="0" --adaptPower --csvPath="/Users/wciezobka/ncn/ftmrate_internal/results/var_power/data/const_dist-${DIST}_pd-${POW}.csv" --ns3Path="/Users/wciezobka/ncn/ns-3-dev/" --mempoolKey="2137"

        # Alternate power
        python3 ml4wifi/envs/ml_wifi_manager.py --scenario="moving" --ml_manager="kf" --managerName="KF" --velocity="0" --startPosition="$DIST" --simulationTime="40" --warmupTime="4" --fuzzTime="2" --measurementsInterval="0.5" --lossModel="Nakagami" --seed="11227" --interval="4" --delta="$POW" --csvPath="/Users/wciezobka/ncn/ftmrate_internal/results/var_power/data/alt_dist-${DIST}_pd-${POW}.csv" --ns3Path="/Users/wciezobka/ncn/ns-3-dev/" --mempoolKey="2137"

        # Adaptation to power
        python3 ml4wifi/envs/ml_wifi_manager.py --scenario="moving" --ml_manager="kf" --managerName="KF" --velocity="0" --startPosition="$DIST" --simulationTime="40" --warmupTime="4" --fuzzTime="2" --measurementsInterval="0.5" --lossModel="Nakagami" --seed="11227" --interval="4" --delta="$POW" --adaptPower --csvPath="/Users/wciezobka/ncn/ftmrate_internal/results/var_power/data/adopt_dist-${DIST}_pd-${POW}.csv" --ns3Path="/Users/wciezobka/ncn/ns-3-dev/" --mempoolKey="2137"

    done
done
