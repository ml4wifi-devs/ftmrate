import re
from glob import glob

import numpy as np
import pandas as pd


def parse_results(fname, sample=None):
    results = []
    rtt_results = []
    regex = (
        r"Target: (([0-9a-f]{2}:*){6}), " +
        r"status: ([0-9]), rtt: ([0-9\-]+) psec, " +
        r"distance: ([0-9\-]+) cm"
    )
    regex_new = (
        r"Target: (([0-9a-f]{2}:*){6}), " +
        r"status: ([0-9]), rtt: ([0-9\-]+) \(  ([0-9\-]+)\) psec, " +
        r"distance: ([0-9\-]+) \(  ([0-9\-]+)\) cm, rssi: ([0-9\-]+) dBm"
    )
    with open(fname) as f:
        data_ori = f.readlines()
    if sample is None:
        data = data_ori
    else:
        data = np.random.choice(data_ori, size=sample, replace=False)
    for line in data:
        match = re.search(regex_new, line)
        if match:
            mac = match.group(1)
            status = int(match.group(3))
            rtt = int(match.group(4))
            rtt_var = int(match.group(5))
            raw_distance = int(match.group(6))
            raw_distance_var = int(match.group(7))
            rssi = int(match.group(8))
        else:
            match = re.search(regex, line)
            if match:
                mac = match.group(1)
                status = int(match.group(3))
                rtt = int(match.group(4))
                raw_distance = int(match.group(5))
            else:
                continue
        if status != 0 or raw_distance < -1000:
            continue
        results.append(raw_distance)
        rtt_results.append(rtt)

    return results


if __name__ == '__main__':
    # Parse measurements into a dict of arrays
    ftm_matrix = []
    for ftm_data in glob('*.out'):

        # Parse one distance to obtain array of ftm measurements
        distance = float(ftm_data.split("_")[0][1:])
        ftm_vals = parse_results(ftm_data)

        for ftm_val in ftm_vals:
            ftm_matrix.append([distance, ftm_val])

    ftm_matrix = np.array(ftm_matrix, dtype=np.float64)
    np.random.shuffle(ftm_matrix)

    ftm_df = pd.DataFrame(ftm_matrix, columns=["distance", "ftmdist"])
    ftm_df.to_csv('ftm.csv', index=None)
