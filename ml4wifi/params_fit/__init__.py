from ml4wifi.params_fit.common import *


# distance and velocity process noise for SDE dynamics in FTM based agents
SIGMA_X = 1.3213624954223633
SIGMA_V = 0.015552043914794922

# exponential smoothing parameters
ES_ALPHA = 0.27773556113243103
ES_BETA = 0.14975710213184357

# Kalman filter distance sensor noise - calculated as the variance of normal distribution which is
# most similar to distance_noise distribution in terms of KL divergence
KF_SENSOR_NOISE = 0.7455304265022278
