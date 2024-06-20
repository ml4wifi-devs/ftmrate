import os
from argparse import ArgumentParser

import pandas as pd


DROP_RUNS = {
    'dynamic': {
        'iwlwifi': [13, 15, 17],
        'ftmrate': [11, 12, 14, 15, 17, 20],
    },
    'static': {
        'iwlwifi': [],
        'ftmrate': [],
    },
}


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('--ftmrate_path', type=str, required=True)
    args.add_argument('--iwlwifi_path', type=str, required=True)
    args.add_argument('--output', type=str, required=True)
    args.add_argument('--runs', type=int, required=True)
    args.add_argument('--type', type=str, required=True)
    args = args.parse_args()

    df = pd.DataFrame(columns=['run', 'device', 'time', 'rate', 'manager'])

    for manager, path in zip(['ftmrate', 'iwlwifi'], [args.ftmrate_path, args.iwlwifi_path]):
        for device in ['sta', 'ap']:
            for run in range(1, args.runs + 1):
                filename = f'{device}_{run}.csv'
                filename = os.path.join(path, filename)

                new_df = pd.DataFrame(columns=['run', 'device', 'time', 'rate', 'manager'])
                source_df = pd.read_csv(filename)

                new_df['time'] = source_df['Time']
                new_df[new_df['time'] == '*REF*'] = "0.0"
                new_df['time'] = new_df['time'].astype(float)
                new_df['rate'] = source_df['Data rate (Mb/s)']
                new_df['run'] = run
                new_df['device'] = device
                new_df['manager'] = manager

                df = pd.concat([df, new_df])

    iwlwifi_drop = DROP_RUNS[args.type]['iwlwifi']
    ftmrate_drop = DROP_RUNS[args.type]['ftmrate']

    df = df[~((df['manager'] == 'iwlwifi') & df['run'].isin(iwlwifi_drop))]
    df = df[~((df['manager'] == 'ftmrate') & df['run'].isin(ftmrate_drop))]

    df.to_csv(args.output, index=False)
