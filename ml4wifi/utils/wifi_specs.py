import numpy as np
import tensorflow_probability.substrates.numpy as tfp

tfb = tfp.bijectors
tfd = tfp.distributions


# LogDistance channel model
# https://www.nsnam.org/docs/models/html/wifi-testing.html#packet-error-rate-performance
# https://www.nsnam.org/docs/models/html/propagation.html#logdistancepropagationlossmodel
DEFAULT_NOISE = -93.97
DEFAULT_TX_POWER = 16.0206
REFERENCE_SNR = DEFAULT_TX_POWER - DEFAULT_NOISE
REFERENCE_LOSS = 46.6777
EXPONENT = 3.0

distance_to_snr_float = lambda distance: REFERENCE_SNR - (REFERENCE_LOSS + 10 * EXPONENT * np.log10(distance))
distance_to_snr = tfb.Shift(REFERENCE_SNR - REFERENCE_LOSS)(tfb.Scale(-10 * EXPONENT / np.log(10.))(tfb.Log()))

# Calculation of SNR uncertainty
SNR_UNCERTAINTY_CONSTANT = 10 * EXPONENT / np.log(10.)


def snr_uncertainty(distance_estimated: float, distance_uncertainty: float) -> float:
    """
    Calculates uncertainty of an estimated SNR. Based on the rule of propagation of uncertainty:
    https://en.wikipedia.org/wiki/Propagation_of_uncertainty#Simplification

    Parameters
    ----------
    distance_estimated : float
        Distance estimated by the agent
    distance_uncertainty : float
        Uncertainty of a distance estimation

    Returns
    -------
    uncertainty : float
        SNR estimation uncertainty
    """

    return (SNR_UNCERTAINTY_CONSTANT / distance_estimated) * distance_uncertainty


# Based on simulation with Nakagami channel model and curve fitting
success_probability_curve_params = np.array([
    [11.65393785513402, 3.8775077976047094, 0.3602975522720451, 1.0966422191036909],
    [11.752459670140652, 3.9483719829596446, 0.33749276122801675, 1.114881614980944],
    [11.91869022606955, 4.1289157328863935, 0.28645357503283975, 1.1992382547701346],
    [12.036155932343735, 4.165430532198794, 0.2667796120885035, 1.2239463918406743],
    [11.711956895426132, 3.9420115043363784, 0.34051397727699184, 1.1093788351445586],
    [15.134034991944615, 3.9536981970248894, 0.3267428835806004, 1.1227637521444083],
    [16.429067127279076, 3.913013069324228, 0.3327343972991688, 1.1200316432734243],
    [17.676273803734627, 3.8738382695249562, 0.3452077084421544, 1.1101468286110574],
    [21.532718477733066, 3.9133803489941137, 0.32874174040037973, 1.129658601444765],
    [23.001409193319798, 3.853182889999326, 0.34588260582895997, 1.1008682211959366],
    [29.507040657141722, 3.81431887300925, 0.3574543188501848, 1.0937818841473395],
    [31.626243272933067, 3.9349669306021546, 0.3157052894363473, 1.143993027709254],
])

# success_probability = tfd.SinhArcsinh(...).cdf(snr)
success_probability_dist = tfd.SinhArcsinh(
    loc=success_probability_curve_params[:, 0],
    scale=success_probability_curve_params[:, 1],
    skewness=success_probability_curve_params[:, 2],
    tailweight=success_probability_curve_params[:, 3],
    distribution=tfd.Normal(0, 1)
)
success_probability = tfb.NormalCDF()(tfb.Invert(success_probability_dist.bijector))


# Based on simulation with LogDistance channel model
wifi_modes_snrs = np.array([
    10.613624240405125,
    10.647249582547907,
    10.660723984151614,
    10.682584060100158,
    11.151267538857537,
    15.413200906170632,
    16.735812667249125,
    18.09117593040658,
    21.80629059204096,
    23.33182497361092,
    29.78890607654747,
    31.750234694079595
])

# success_probability = tfd.Normal(wifi_modes_snrs, 1 / (2 * sqrt(2)).cdf(snr)
success_probability_log_distance = tfb.NormalCDF()(tfb.Scale(1 / (2 * np.sqrt(2)))(tfb.Shift(-wifi_modes_snrs)))


wifi_modes_rates = np.array([
    7.3,
    14.6,
    21.9,
    29.3,
    43.9,
    58.5,
    65.8,
    73.1,
    87.8,
    97.5,
    109.7,
    121.9
])

# success_probability_to_rate = wifi_modes_rates * success_probability
success_probability_to_rate = tfb.Scale(wifi_modes_rates)


def expected_rates(tx_power: float):
    return tfb.Chain([
        success_probability_to_rate,
        success_probability,
        tfb.Shift(tx_power - DEFAULT_NOISE - REFERENCE_LOSS)(tfb.Scale(-10 * EXPONENT / np.log(10.))(tfb.Log()))
    ])


def expected_rates_log_distance(tx_power: float):
    return tfb.Chain([
        success_probability_to_rate,
        success_probability_log_distance,
        tfb.Shift(tx_power - DEFAULT_NOISE - REFERENCE_LOSS)(tfb.Scale(-10 * EXPONENT / np.log(10.))(tfb.Log()))
    ])


def ideal_mcs(distance: float, tx_power: float) -> np.int32:
    return np.argmax(expected_rates(tx_power)(distance))


def ideal_mcs_log_distance(distance: float, tx_power: float) -> np.int32:
    return np.argmax(expected_rates_log_distance(tx_power)(distance))


# FTM distance measurement noise model (fig. 3):
# https://www2.tkn.tu-berlin.de/bib/zubow2022ftm-ns3/zubow2022ftm-ns3.pdf
RTT_LOC = -5478.0   # [ps]
RTT_SCALE = 2821.0  # [ps]
RTT_LAMBDA = 0.000183
RTT_TO_D = 0.00015

distance_noise = tfb.Scale(RTT_TO_D)(tfd.ExponentiallyModifiedGaussian(
    loc=RTT_LOC,
    scale=RTT_SCALE,
    rate=RTT_LAMBDA
))
