from argparse import ArgumentParser

import pandas as pd


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('--a', required=True, type=float)
    args.add_argument('--b', required=True, type=float)
    args = args.parse_args()

    df = pd.read_csv('ftm.csv')

    corrected_ftm = (df['ftmdist'] - args.b) / args.a
    errors = df['distance'] - corrected_ftm
    noise = (errors ** 2).mean()

    print(f'KF sensor noise: {noise}')
