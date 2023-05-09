import os

import pandas as pd


CSV_FILES_DIR = 'csv_files'
PARAMS_HEADER = 'mcs,loc,scale,skewness,tailweight,es_alpha,es_beta,kf_sensor_noise,sigma_x,sigma_v\n'


def load_parameters_file(name: str = 'parameters.csv'):
    path = os.path.join(CSV_FILES_DIR, name)

    if not os.path.isfile(path):
        file = open(path, 'w')
        file.write(PARAMS_HEADER)
        for mcs in range(12):
            file.write(f'{mcs}' + ',' * 9 + '\n')
        file.close()
    
    return pd.read_csv(path)
