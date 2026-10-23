import matplotlib.pyplot as plt
import numpy as np
from radvel.kepler import rv_drive
from scipy.stats import moment

"""
Calculate the standard error on the standard deviation of the RV predictions
of a Keplerian function, assuming uniform time sampling and iid observations.
NOTE: this does not assume the stddevs are normally distributed!

AI DISCLOSURE: I used Claude to calculate the formulas here, and I coded this script by hand to check them.
"""


def kep_var(per, Tp, ecc, omega_st, K):
    beta = np.sqrt(1 - ecc**2)
    kep_var = (
        K**2
        * beta**2
        * (np.sin(omega_st) ** 2 + beta * np.cos(omega_st) ** 2)
        / (1 + beta)
    )
    return kep_var


def kep_mu4(per, Tp, ecc, omega_st, K):
    c = np.cos(omega_st)
    s = np.sin(omega_st)
    beta = np.sqrt(1 - ecc**2)
    kep_mu4 = (
        K**4
        * 3
        * beta**3
        / (2 * (1 + beta) ** 2)
        * (
            (1 + 2 * beta - 2 * beta**3) * c**4
            + 2 * (1 + 2 * beta - 2 * beta**2) * c**2 * s**2
            + s**4
        )
    )
    return kep_mu4


def std_err_of_stddev(per, Tp, ecc, omega_st, K, num_samples):

    sigma2 = kep_var(per, Tp, ecc, omega_st, K)
    mu4 = kep_mu4(per, Tp, ecc, omega_st, K)

    var_of_stddev = ((num_samples - 1) * mu4 - (num_samples - 3) * sigma2**2) / (
        4 * num_samples**2 * sigma2
    )
    std_err_of_stddev = np.sqrt(var_of_stddev)

    return std_err_of_stddev


if __name__ == "__main__":
    # some unit tests; do a check for each formula
    fig, ax = plt.subplots(2, 1)
    per = 365  # [d]
    K = 5  # [m/s]
    omega_st = np.pi / 2  # [rad]
    Tp = 0  # [d]
    num_samples_to_test = [int(1e3), int(1e4), int(1e5), int(1e6), int(1e7)]
    ecc_to_test = [0, 0.5, 0.9]
    for i, num_samples in enumerate(num_samples_to_test):
        for j, ecc in enumerate(ecc_to_test):
            epochs = np.random.uniform(0, per, num_samples)

            orbital_params = np.array([per, Tp, ecc, omega_st, K])

            rvs = rv_drive(epochs, orbital_params)
            rv_var = np.var(rvs)
            rv_mu4 = moment(rvs, order=4)
            label = None
            if i == j == 0:
                label = "variance"
            ax[0].scatter(
                num_samples,
                (rv_var - kep_var(per, Tp, ecc, omega_st, K)) / rv_var,
                color="grey",
                alpha=0.5,
                label=label,
            )
            if i == j == 0:
                label = "mu4"
            ax[0].scatter(
                num_samples,
                (rv_mu4 - kep_mu4(per, Tp, ecc, omega_st, K)) / rv_mu4,
                color="purple",
                alpha=0.5,
                label=label,
            )

    print("Panel 1 done!")
    num_observations = 20
    num_iterations = [
        int(1e2),
        int(1e3),
        int(1e4),
        int(1e5),
        # int(1e6),
    ]  # the number of times we imagine we measure the stddev of the Keplerian metric

    empirical_stderrors = np.empty((len(ecc_to_test), len(num_iterations)))
    for i, ecc in enumerate(ecc_to_test):
        for j, n_iter in enumerate(num_iterations):
            kep_stddevs = np.empty(n_iter)
            for k in range(n_iter):
                epochs = np.random.uniform(0, per, num_observations)

                orbital_params = np.array([per, Tp, ecc, omega_st, K])

                rvs = rv_drive(epochs, orbital_params)
                kep_stddevs[k] = np.std(rvs)
            empirical_stderrors[i, j] = np.std(kep_stddevs)
            ax[1].scatter(
                n_iter,
                (
                    empirical_stderrors[i, j]
                    - std_err_of_stddev(per, Tp, ecc, omega_st, K, num_observations)
                )
                / empirical_stderrors[i, j],
                color="purple",
                alpha=0.5,
            )
    print("Panel 2 done!")
    ax[0].legend()
    ax[0].set_xlabel("Num. samples")
    ax[1].set_xlabel("Num. samples of Keplerian stddev")
    ax[0].set_title("fractional error on variance and $\\mu_4$")
    ax[1].set_title(
        f"fractional error on standard error of the stddev ({num_observations} obs)"
    )
    ax[0].set_ylabel(
        "(emprical - analytic)/empirical"
    )  # this should converge to 0 as num_samples increases and the empirical estimate improves
    ax[1].set_ylabel(
        "(emprical - analytic)/empirical"
    )  # this should converge to 0 as num_samples increases and the empirical estimate improves

    for a in ax:
        a.axhline(0, ls="--", color="k", alpha=0.1)
        a.set_xscale("log")
    plt.tight_layout()
    plt.savefig("kep_moments_check.png", dpi=250)
