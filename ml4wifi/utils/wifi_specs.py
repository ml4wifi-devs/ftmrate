import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
from chex import Scalar

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

distance_to_snr_scalar = lambda distance: REFERENCE_SNR - (REFERENCE_LOSS + 10 * EXPONENT * jnp.log10(distance))
distance_to_snr = tfb.Shift(REFERENCE_SNR - REFERENCE_LOSS)(tfb.Scale(-10 * EXPONENT / jnp.log(10.))(tfb.Log()))

# Calculation of SNR uncertainty
SNR_UNCERTAINTY_CONSTANT = 10 * EXPONENT / jnp.log(10.)


@jax.jit
def snr_uncertainty(distance_estimated: Scalar, distance_uncertainty: Scalar) -> Scalar:
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
success_probability_curve_params = jnp.array([
    [11.808264635318604, 3.969739514764798, 0.32612954734945415, 1.1399195015627401],
    [11.89485137165589, 4.085207922809933, 0.2886292631721735, 1.2497610371821313],
    [11.589187177412986, 3.8680505263566083, 0.3711375055053177, 1.0759003814091108],
    [11.209270798782597, 3.645803021518187, 0.46718303793099125, 1.0179815116735462],
    [11.592330125549454, 3.8334098671797756, 0.37839033304222275, 1.0743442113148782],
    [15.280161386960096, 4.034025052989751, 0.2931889537363139, 1.1467356646987326],
    [16.486683986358468, 3.965258460083516, 0.31669257468987655, 1.146802419509176],
    [17.5283573281527, 3.7633338278423647, 0.3849448682520248, 1.062220507445853],
    [21.454031767294953, 3.8483936731647237, 0.34798706629623066, 1.1011861008676405],
    [23.079014584329915, 3.9068810663208984, 0.32581084830275386, 1.1305465779866442],
    [29.79588258681244, 4.032009013660586, 0.2851674101793301, 1.1849286990844083],
    [31.521294079499764, 3.868334574332342, 0.3394212105587342, 1.1193924247016405]
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
wifi_modes_snrs = jnp.array([
    -0.0105,
    2.92567,
    6.04673,
    8.98308,
    12.5948,
    16.4275,
    17.9046,
    19.6119,
    23.5752,
    24.8097,
    31.2291,
    33.1907,
])

# success_probability = tfd.Normal(wifi_modes_snrs, 1 / (2 * sqrt(2)).cdf(snr)
success_probability_log_distance = tfb.NormalCDF()(tfb.Scale(1 / (2 * jnp.sqrt(2)))(tfb.Shift(-wifi_modes_snrs)))


wifi_modes_rates = jnp.array([
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


expected_rates = tfb.Chain([success_probability_to_rate, success_probability, distance_to_snr])
expected_rates_log_distance = tfb.Chain([success_probability_to_rate, success_probability_log_distance, distance_to_snr])


@jax.jit
def ideal_mcs(distance: Scalar) -> jnp.int32:
    return jnp.argmax(expected_rates(distance))


@jax.jit
def ideal_mcs_log_distance(distance: Scalar) -> jnp.int32:
    return jnp.argmax(expected_rates_log_distance(distance))


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
