from ml4wifi.params_fit.common import *


# distance and velocity process noise for SDE dynamics in FTM based agents
SIGMA_R = 0.83820117
SIGMA_V = 0.3324591

# exponential smoothing parameters
ES_ALPHA = 0.27773556113243103
ES_BETA = 0.14975710213184357

# Kalman filter distance sensor noise - calculated as the variance of normal distribution which is
# most similar to distance_noise distribution in terms of KL divergence
KF_SENSOR_NOISE = 0.7454228401184082
