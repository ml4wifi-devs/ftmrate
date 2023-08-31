import pandas as pd

from tools.plots.common import ALL_MANAGERS, DATA_FILE, get_thr_ci


MAX_N_WIFI = 16


def print_results(distance: float) -> None:
    df = pd.read_csv(DATA_FILE)
    df = df[(df.mobility == 'Distance') & (df.distance == distance)]
    df = df[df.nWifi == df.nWifiReal]

    oracle_mean, _, _ = get_thr_ci(df[df.manager == 'Oracle'], 'nWifiReal')

    for manager in list(ALL_MANAGERS.keys()) + ['OracleFTM']:
        mean, _, _ = get_thr_ci(df[df.manager == manager], 'nWifiReal')
        difference = (oracle_mean - mean) / oracle_mean * 100

        print(f'{manager}: {difference}\n')


if __name__ == '__main__':
    print_results(distance=1)
