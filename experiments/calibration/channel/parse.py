from glob import glob

import numpy as np
import pandas as pd


if __name__ == '__main__':
    # Parse measurements into a dict of arrays
    rssi_matrix = []

    for rssi_data in glob('*.out'):
        # Parse one distance to obtain array of rssi measurements
        distance = float(rssi_data.split("_")[0][1:])
        rssi_file = open(rssi_data, "r")
        for line in rssi_file.readlines():

            # print(len(line.split(" ")))
            if len(line.split(" ")) == 20:
                rssi_val = line.split(" ")[4][:-3]
                rssi_matrix.append([distance, rssi_val])

    rssi_matrix = np.array(rssi_matrix, dtype=np.float64)
    np.random.shuffle(rssi_matrix)

    rssi_df = pd.DataFrame(rssi_matrix, columns=["distance", "rssi"])
    rssi_df.to_csv('rssi.csv', index=None)
