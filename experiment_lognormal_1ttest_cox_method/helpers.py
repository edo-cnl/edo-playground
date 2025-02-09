import numpy as np
import scipy.stats
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

def pull_observations_from_distribution(sample_size: int, is_lognormal = True, population_mean = 0.0, population_st_dev = 1.0):
    """
    Generates a sample from a distribution.

    Args:
        sample_size: The number of observations in the sample.
        is_lognormal: Boolean indicating whether to generate observations from a lognormal distribution.

    Returns:
        A NumPy array containing the generated sample.
    """
    if is_lognormal: 
        return rng.lognormal(mean=population_mean, sigma=population_st_dev, size=sample_size)
    return np.random.normal(loc=population_mean, scale=population_st_dev, size=sample_size)


def one_tail_test_lognormal_distribution_modified_cox(sample_observations: np.array, comparison_value: float, alternative = 'sample_mean_lower', alpha = 0.05, bootstrap_replicates = 0):
    """
    One-tail test for lognormal distribution
        sample_observations: list of lognormally distributed observations
        comparison_value: comparison value
        alternative: side of the distribution to test
        alpha: significance level
        bootstrap_replicates: number of bootstrap replicates for the sample standard deviation estimation
    :return: boolean value indicating whether the null hypothesis is rejected
    """
    # Check for non-positive values in sample_observations
    if np.any(sample_observations <= 0):
        raise ValueError('sample_observations contains non-positive values')
    
    if alternative not in ['sample_mean_lower', 'sample_mean_greater']:
        raise ValueError('alternative must be "sample_mean_lower" or "sample_mean_greater"')

    log_observations = np.log(sample_observations)
    log_mean = np.mean(log_observations)
    sample_size = len(sample_observations)
    
    if bootstrap_replicates > 0:
        bootstrap_std_devs = []
        for _ in range(bootstrap_replicates):
            bootstrap_sample = np.random.choice(log_observations, size=sample_size, replace=True)
            bootstrap_std_devs.append(bootstrap_sample.std())
        log_st_dev = np.mean(bootstrap_std_devs)
    elif bootstrap_replicates == 0:
        log_st_dev = np.std(log_observations)
    else:
        raise ValueError('bootstrap_replicates must be a non-negative integer')

    inverted_mean = log_mean + log_st_dev**2 / 2

    inverted_margin_of_error = (
        scipy.stats.t.ppf(1 - alpha, sample_size - 1)
        * (
            (log_st_dev**2 / sample_size)
            + (log_st_dev**4 / (2 * (sample_size - 1)))        )
        ** 0.5
    )


    if alternative == 'sample_mean_lower':
        inverted_upper_bound = np.exp(
            inverted_mean + inverted_margin_of_error
        )

        return inverted_upper_bound < comparison_value

    elif alternative == 'sample_mean_greater':
        inverted_lower_bound = np.exp(
            inverted_mean - inverted_margin_of_error
        )

        return inverted_lower_bound > comparison_value
    
def run_one_test(sample_size, is_lognormal=True, normal_mean=0, normal_st_dev=1, alpha=0.05, bootstrap_replicates = 1):
    sample_observations = pull_observations_from_distribution(sample_size, is_lognormal, normal_mean, normal_st_dev)
    lognormal_mean = np.exp(normal_mean + normal_st_dev**2 / 2)
    null_hypothesis_rejected = one_tail_test_lognormal_distribution_modified_cox(sample_observations, lognormal_mean, alternative='sample_mean_lower', alpha=alpha / 2, bootstrap_replicates=bootstrap_replicates) or one_tail_test_lognormal_distribution_modified_cox(sample_observations, lognormal_mean, alternative='sample_mean_greater', alpha=alpha / 2, bootstrap_replicates=bootstrap_replicates)
    return null_hypothesis_rejected

def run_experiment(
        sample_sizes: list[int] 
        , significance_level = 0.05
        , sample_replications = 1
        , normal_mean = 0
        , normal_st_dev = 1
        , bootstrap_replicates = 0):
    """
    Run experiment for two-tails test for lognormal distribution
        start_sample_size: starting sample size
        end_sample_size: ending sample size
        increase_in_sample_size: increase in sample size
        sample_replications: number of sample replications
        bootstrap_replicates: number of bootstrap replicates for the sample standard deviation estimation
        significance_level: significance level
    :return: list of boolean values indicating whether the null hypothesis is rejected
    """
    

    df = pd.DataFrame()
    index = 0

    if len(sample_sizes) > 0:
        for i in sample_sizes:
            for j in range(sample_replications):
                null_hypothesis_rejected = run_one_test(sample_size=i, is_lognormal=True, normal_mean=normal_mean, normal_st_dev=normal_st_dev, alpha=significance_level, bootstrap_replicates=bootstrap_replicates)
                tdf = pd.DataFrame({'sample_size': i, 'null_hypothesis_rejected': null_hypothesis_rejected}, index=[i])
                df = pd.concat([df, tdf])
                index += 1
        return df        