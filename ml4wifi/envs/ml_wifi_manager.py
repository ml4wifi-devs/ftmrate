#!/usr/bin/python3

import os
import pathlib
from argparse import ArgumentParser, ArgumentError

from py_interface import *

from ml4wifi.agents.exponential_smoothing import ManagersContainer as ExponentialSmoothing
from ml4wifi.agents.kalman_filter import ManagersContainer as KalmanFilter
from ml4wifi.agents.particle_filter import ManagersContainer as ParticleFilter
from ml4wifi.agents.thompson_sampling import ManagersContainer as ThompsonSampling
from ml4wifi.agents.hybrid_threshold import ManagersContainer as HybridThreshold
from ml4wifi.agents.hybrid_mab import ManagersContainer as HybridMAB

from ml4wifi.envs.ns3_ai_structures import Env, Act


# Managers dict
MANAGERS = {
    'es': ExponentialSmoothing,
    'kf': KalmanFilter,
    'pf': ParticleFilter,
    'ts': ThompsonSampling,
    'thr': HybridThreshold,
    'mab': HybridMAB
}

# Simulation parameters for different scenarios
NS3_ARGS = {
    'manager': 'ns3::MlWifiManager'
}


def main() -> None:
    global MANAGERS, NS3_ARGS

    # Program arguments
    parser = ArgumentParser()

    # Environment parameters
    parser.add_argument('--ml_manager', required=True, type=str)
    parser.add_argument('--seed', default=100, type=int)
    parser.add_argument('--mempoolKey', default=1234, type=int)
    parser.add_argument('--ns3Path', default=f'{pathlib.Path.home()}/ns-3-dev/', type=str)

    # ns-3 cmd arguments
    parser.add_argument('--ampdu', default=True, action='store_false')
    parser.add_argument('--area', default=40., type=float)
    parser.add_argument('--channelWidth', default=20, type=int)
    parser.add_argument('--csvPath', default='results.csv', type=str)
    parser.add_argument('--dataRate', default=125, type=int)
    parser.add_argument('--delta', default=0, type=float)
    parser.add_argument('--distance', default=0., type=float)
    parser.add_argument('--enableRtsCts', default='False', type=str)
    parser.add_argument('--fuzzTime', default=5., type=float)
    parser.add_argument('--interval', default=2, type=float)
    parser.add_argument('--lossModel', default='Nakagami', type=str)
    parser.add_argument('--managerName', type=str)
    parser.add_argument('--mcs', type=int)
    parser.add_argument('--minGI', default=3200, type=int)
    parser.add_argument('--measurementsInterval', default=1., type=float)
    parser.add_argument('--nodePause', default=20., type=float)
    parser.add_argument('--nodeSpeed', default=1.4, type=float)
    parser.add_argument('--nWifi', default=1, type=int)
    parser.add_argument('--packetSize', default=1500, type=int)
    parser.add_argument('--pcapName', type=str)
    parser.add_argument('--scenario', required=True, type=str)
    parser.add_argument('--simulationTime', default=20., type=float)
    parser.add_argument('--startPosition', default=0., type=float)
    parser.add_argument('--velocity', default=1., type=float)
    parser.add_argument('--wallInterval', default=0., type=float)
    parser.add_argument('--wallLoss', default=3., type=float)
    parser.add_argument('--warmupTime', default=10., type=int)

    # Other arguments
    parser.add_argument('--main_retransmissions', default=2, type=int)
    parser.add_argument('--backup_retransmissions', default=2, type=int)
    parser.add_argument('--historyLength', default=25, type=int)
    parser.add_argument('--threshold', default=0.7, type=float)
    parser.add_argument('--mab_decay', default=1.0, type=float)
    parser.add_argument('--verbose', default=False, action='store_true')

    args = parser.parse_args()

    # Adjust the ns-3 path
    if os.environ.get('NS3_DIR'):
        args.ns3Path = os.environ.get('NS3_DIR')


    # Set manager type
    if '_' in args.ml_manager:
        ml_manager, ftmrate_agent = args.ml_manager.split('_')
        if ml_manager == 'thr':
            kwargs = dict(
                backup_retransmissions=args.backup_retransmissions,
                main_retransmissions=args.main_retransmissions,
                ftmrate_agent=ftmrate_agent,
                history_length=args.historyLength,
                threshold=args.threshold
            )
        elif ml_manager == 'mab':
            kwargs = dict(
                ftmrate_agent=ftmrate_agent,
                mab_decay=args.mab_decay
            )
        else:
            raise ArgumentError(None, f"'{ml_manager}' is not in available as a hybrid approach. Choose 'thr' or 'mab'")
    else:
        ml_manager = args.ml_manager
        kwargs = dict()

    if ml_manager not in MANAGERS:
        raise ArgumentError(None, f"'{ml_manager}' is not in available managers set")

    managers_container = MANAGERS[ml_manager](args.seed, **kwargs)


    # Set common arguments
    NS3_ARGS['RngRun'] = args.seed
    NS3_ARGS['ampdu'] = args.ampdu
    NS3_ARGS['channelWidth'] = args.channelWidth
    NS3_ARGS['csvPath'] = args.csvPath
    NS3_ARGS['dataRate'] = args.dataRate
    NS3_ARGS['delta'] = args.delta
    NS3_ARGS['enableRtsCts'] = True if args.enableRtsCts == 'True' else False
    NS3_ARGS['fuzzTime'] = args.fuzzTime
    NS3_ARGS['interval'] = args.interval
    NS3_ARGS['lossModel'] = args.lossModel
    NS3_ARGS['managerName'] = args.managerName if args.managerName else ml_manager
    NS3_ARGS['minGI'] = args.minGI
    NS3_ARGS['packetSize'] = args.packetSize
    NS3_ARGS['simulationTime'] = args.simulationTime
    NS3_ARGS['warmupTime'] = args.warmupTime

    if args.pcapName:
        NS3_ARGS['pcapName'] = args.pcapName


    # Set args according to scenario
    if args.scenario == 'distance':
        pname = 'stations'
        NS3_ARGS['distance'] = args.distance
        NS3_ARGS['mobilityModel'] = 'Distance'
        NS3_ARGS['nWifi'] = args.nWifi

    elif args.scenario == 'rwpm':
        pname = 'stations'
        NS3_ARGS['area'] = args.area
        NS3_ARGS['mobilityModel'] = 'RWPM'
        NS3_ARGS['nodePause'] = args.nodePause
        NS3_ARGS['nodeSpeed'] = args.nodeSpeed
        NS3_ARGS['nWifi'] = args.nWifi

    elif args.scenario == 'moving':
        pname = 'moving'
        NS3_ARGS['measurementsInterval'] = args.measurementsInterval
        NS3_ARGS['startPosition'] = args.startPosition
        NS3_ARGS['velocity'] = args.velocity
        NS3_ARGS['wallInterval'] = args.wallInterval
        NS3_ARGS['wallLoss'] = args.wallLoss

    elif args.scenario == 'hidden':
        pname = 'stations'
        NS3_ARGS['distance'] = args.distance
        NS3_ARGS['mobilityModel'] = 'Hidden'
        NS3_ARGS['nWifi'] = args.nWifi

    else:
        raise ArgumentError(None, 'Bad scenario selected')


    # Show all ns3 args if verbose flag on
    if args.verbose:
        print(f'\nNs3 args:')
        for key, val in sorted(NS3_ARGS.items(), key=lambda x: x[0]):
            print(f'  {key}: {val}')


    # Shared memory settings
    memblock_key = 2333
    mempool_key = args.mempoolKey
    mem_size = 256


    # Initialize ns3-ai
    exp = Experiment(mempool_key, mem_size, pname, args.ns3Path, debug=False)
    var = Ns3AIRL(memblock_key, Env, Act)


    # Run ns-3
    try:
        ns3_process = exp.run(setting=NS3_ARGS, show_output=True)

        while not var.isFinish():
            with var as data:
                if data is None:
                    break

                data.act = managers_container.do(data.env, data.act)

        ns3_process.wait()
    finally:
        del exp


if __name__ == '__main__':
    main()
