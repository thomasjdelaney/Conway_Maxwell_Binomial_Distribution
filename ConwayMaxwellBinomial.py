"""
For the class of ConwayMaxwellBinomial distribution object and all useful functions relating to it.
"""
import numpy as np
from scipy.special import comb, logit
from scipy.optimize import minimize

class ConwayMaxwellBinomial(object):
    def __init__(self, p, nu, m):
        """
        Creates the Conway-Maxwell binomial distribution with parameters p, nu, and m. Calculates the normalising function during initialisation. Uses exponents and logs to avoid overflow.
        Arguments:  self,
                    p, real 0 <= p <= 1, probability of success
                    nu, real, dispersion parameter
                    m, number of trials
        Returns:    object
        """
        self.p = p
        self.nu = nu
        self.m = m
        self.normaliser = self.getNormaliser()
        self.has_samp_des_dict = False
        self.samp_des_dict, self.has_samp_des_dict = self.getSamplingDesignDict()

    def pmf_atomic(self, k):
        """
        Probability mass function. Uses exponents and logs to avoid overflow.
        Arguments:  self, ConwayMaxwellBinomial object,
                    k, int, must be an integer in the interval [0, m]
        Returns:    P(k)
        """
        if (k > self.m) | (k != int(k)) | (k < 0):
            raise ValueError("k must be an integer between 0 and m, inclusive")
        if self.p == 1:
            p_k = 1 if k == self.m else 0
        elif self.p == 0:
            p_k = 1 if k == 0 else 0
        elif self.has_samp_des_dict:
            p_k = self.samp_des_dict.get(k)
        else:
            p_k = self.getProbMassForCount(k)/self.normaliser
        return p_k

    def pmf(self, k):
        """
        Probability mass function that can take lists or atomics.
        Arguments:  self, ConwayMaxwellBinomial object,
                    k, int, or list of ints
        Returns:    P(k)
        """
        if np.isscalar(k):
            return self.pmf_atomic(k)
        else:
            return np.array([self.pmf_atomic(k_i) for k_i in k])

    def logpmf(self, k):
        """
        Log probability mass function. Does what it says on the tin. 
        Improvement might be possible, later.
        Arguments:  self, ConwayMaxwellBinomial object,
                    k, int, must be an integer in the interval [0,m]
        Returns:    log P(k)
        """
        return np.log(self.pmf(k))

    def cdf_atomic(self, k):
        """
        For getting the cumulative distribution function of the distribution at k.
        Arguments:  self, the distribution object
                    k, int, must be an integer in the interval [0,m]
        Returns:    float

        NB: this function relies on the sampling design dictionary keys being sorted!
        """
        accumulated_density = 0
        if (k > self.m) | (k != int(k)) | (k < 0):
            raise ValueError("k must be an integer between 0 and m, inclusive")
        elif k == 0:
            return self.samp_des_dict[0]
        elif k == self.m:
            return 1.0
        else:
            for dk,dv in self.samp_des_dict.items():
                if dk <= k:
                    accumulated_density += dv
                else:
                    return accumulated_density # avoids looping through all the keys unnecessarily.
    
    def cdf(self, k):
        """
        For getting the cumulative distribution function at k, or a list of k.
        Arguments:  self, the distribution object
                    k, int, must be an integer in the interval [0,m]
        Returns:    float or array of floats
        """
        if np.isscalar(k):
            return self.cdf_atomic(k)
        else:
            return np.array([self.cdf_atomic(k_i) for k_i in k])

    def getSamplingDesignDict(self):
        """
        Returns a dictionary representing the sampling design of the distribution. That is, samp_des_dict[k] = pmf(k)
        Arguments:  self, the distribution object,
        Returns:    samp_des_dict, dictionary, int => float
                    has_samp_des_dict, True
        """
        possible_values = range(0,self.m+1)
        samp_des_dict = dict(zip(possible_values, self.pmf(possible_values)))
        has_samp_des_dict = True
        return samp_des_dict, has_samp_des_dict

    def rvs(self, size=1):
        return np.random.choice(range(0,self.m + 1), size=size, replace=True, p=list(self.samp_des_dict.values()))

    def getNormaliser(self):
        """
        For calculating the normalising factor of the distribution.
        Arguments:  self, the distribution object
        Returns:    the value of the normalising factor S(p,nu)
        """
        if (self.p == 0) | (self.p == 1):
            warnings.warn("p = " + str(self.p) + " The distribution is deterministic.")
            return 0
        else:
            return np.sum([self.getProbMassForCount(k) for k in range(0, self.m + 1)])

    def getProbMassForCount(self, k):
        """
        For calculating the unnormalised probability mass for an individual count.
        Arguments:  self, the distribution object
                    k, int, must be an integer in the interval [0, m]
        Returns:    float, 
        """
        return np.exp((self.nu * np.log(comb(self.m, k))) + (k * np.log(self.p)) + ((self.m - k) * np.log(1-self.p)))

def getLogFactorial(k):
    """
    For calculating log(k!) = log(k) + log(k-1) + ... + log(2) + log(1).
    Arguments:  k, int
    Returns:    log(k!)
    """
    return np.log(range(1,k+1)).sum() if k else 0.0

def getSecondHyperparam(m):
    """
    Return a value for b that will centre the prior on nu=1.
    """
    numerator = -np.sum([calculateSecondSufficientStat(k,m) * comb(m,k) for k in range(0,m+1)])
    return numerator/(2**m)

def calculateSecondSufficientStat(samples,m):
    """
    For calculating the second sufficient stat for the conway maxwell binomial distribution. k!(m-k)!
    if samples is an array, sums over the samples.
    Arguments:  samples, integer or array of integers
                m, the maximum value of any given sample
    Returns:    \sum_{i=1}^n k_i! (m - k_i)! where k_i is a sample
    """
    samples = np.array([samples]) if np.isscalar(samples) else samples
    return np.sum([getLogFactorial(k) + getLogFactorial(m - k) for k in samples])

def calcUpperBound(a,c,m):
    """
    Helper function for conjugateProprietyTest.
    """
    ratio=a/c
    floored_ratio = np.floor(ratio).astype(int)
    ceiled_ratio = np.ceil(ratio).astype(int)
    return -calculateSecondSufficientStat(floored_ratio,m) + (ratio - floored_ratio)*(-calculateSecondSufficientStat(ceiled_ratio,m) + calculateSecondSufficientStat(floored_ratio,m))

def conjugateProprietyTest(a,b,c,m):
    """
    The conjugate posterior of the Conway-Maxwell binomial distribution is only proper for certain values of the hyperparameters.
    This function tests if the hyperparameters are within these values.
    Arguments:  a, float, the first hyperparameter, corresponds to the first sufficient statistic \sum k_i
                b, float, the second hyperparameter, corresponds to the seconds sufficient statistic, \sum log(k_i!(m-k_i)!)
                c, int, the pseudocount hyperparameter.
    Returns:    None, or raises an error
    """
    assert 0 < (a/c), "a/c <= 0"
    assert (a/c) < m, "a/c >= m"
    assert -getLogFactorial(m) < (b/c), "-log(m!) >= b/c"
    assert (b/c) < calcUpperBound(a,c,m), "(b/c) > t(floor(a/c)) + (a/c - floor(a/c))(t(ceil(a/c)) - t(floor(a/c)))"
    return None

def conwayMaxwellBinomialPriorKernel(com_params, a, b, c, m):
    """
    For calculating the kernel of the conjugate prior of the Conway-Maxwell binomial distribution. 
    Arguments:  com_params, p, nu, the parameters of the Conway-Maxwell binomial distribution
                a, hyperparameter corresponding to the first sufficient stat,
                b, hyperparameter corresponding to the second sufficient stat,
                c, hyperparameter corresponding to the pseudocount
                m, int, the number of bernoulli variables, considered fixed and known
    Returns:    The value of the kernel of the conjugate prior 
    """
    conjugateProprietyTest(a,b,c,m)
    # propriety_dist = norm(0, 1)
    p, nu = com_params
    if (p == 1) | (p == 0):
        return 0
    test_dist = ConwayMaxwellBinomial(p, nu, m)
    natural_params = np.array([logit(p), nu])
    pseudodata_part = np.dot(natural_params, np.array([a,b]))
    partition_part = np.log(test_dist.normaliser) - (nu * getLogFactorial(m)) - (m * np.log(1-p))
    # propriety_part = norm.pdf(logit(p)) * norm.pdf(nu - 1)
    return np.exp(pseudodata_part - c * partition_part)

def conwayMaxwellBinomialPosteriorKernel(com_params, a, b, c, suff_stats, m, n):
    """
    For calculating the kernel of the posterior distribution of a Conway-Maxwell binomial distribution at parameter values 'params'.
    Parameters are assumed to be in canonical form, rather than natural.
    Arguments:  com_params, 2 element 1-d numpy array (float), the parameter values for the Conway-Maxwell binomial distribution
                a, hyperparameter corresponding to the first sufficient stat,
                b, hyperparameter corresponding to the second sufficient stat,
                c, hyperparameter corresponding to the pseudocount
                suff_stats, 2 element array, sufficient statistics of the Conway-Maxwell binomial distribution, calculated from data, (sum(k_i), sum(log(k_i!(m-k_i!))))
                m, int, the number of bernoulli variables, considered fixed and known.
                n, int, number of data points
    Returns: the kernel value at (p, nu) = params
    """
    # propriety_dist = norm(0, 1)
    conjugateProprietyTest(a,b,c,m)
    p, nu = com_params
    if (p == 1) | (p == 0):
        return 0
    chi = np.array([a, b])
    natural_params = np.array([logit(p), nu])
    data_part = np.dot(natural_params, chi + suff_stats)
    test_dist = ConwayMaxwellBinomial(p, nu, m)
    partition_part = np.log(test_dist.normaliser) - (nu * getLogFactorial(m)) - (m * np.log(1-p))
    # propriety_part = norm.pdf(logit(p)) * norm.pdf(nu - 1)
    total_count = n + c # includes pseudocounts
    return np.exp(data_part - total_count * partition_part)

def conwayMaxwellNegLogLike(params, m, samples):
    """
    For calculating the negative log likelihood at p,nu.
    Arguments:  params: p, 0 <= p <= 1
                        nu, float, dispersion parameter
                m, number of bernoulli variables
                samples, ints between 0 and m, data.
    Returns:    float, negative log likelihood
    """
    p, nu = params
    if (p == 1) | (p == 0):
        return np.infty
    n = samples.size
    com_dist = ConwayMaxwellBinomial(p, nu, m)
    p_part = np.log(p/(1-p))*samples.sum()
    nu_part = nu * np.log(comb(m,samples)).sum()
    partition_part = np.log(com_dist.normaliser) - (m * np.log(1-p))
    return n*partition_part - p_part - nu_part

def estimateParams(m, samples, init):
    """
    For estimating the parameters of the Conway-Maxwell binomial distribution from the given samples.
    Arguments:  m, the number of bernoulli variables being used.
                samples, ints, between 0 and m
    Return:     the fitted params, p and nu
    """
    bnds = ((np.finfo(float).resolution, 1 - np.finfo(float).resolution),(None,None))
    res = minimize(conwayMaxwellNegLogLike, init, args=(m,samples), bounds=bnds)
    return res.x

