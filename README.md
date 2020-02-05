# Conway Maxwell Binomial Distribution
 This package defines a class of Conway-Maxwell-Binomial distribution objects for Python 3. It also defines some useful functions for working with the objects.

#### Requirements
- Python3.6 or above
- numpy
- scipy

## Conway-Maxwell-Binomial distributions
 If you would like to learn more about this probability distribution, see this article: <https://en.wikipedia.org/wiki/Conway-Maxwell-binomial_distribution>

 Or this paper: <https://arxiv.org/pdf/1404.1856v1.pdf>

## Initialising a COMB distribution
 To initialse a Conway-Maxwell-Binomial distribution use the following lines of code
 ```python
 import ConwayMaxwellBinomial as cmb
 p = 0.4
 nu = 0.9
 m = 100
 com_distn = cmb.ConwayMaxwellBinomial(p, nu, m)
 ```
 To sample from the distribution run
 ```python
 com_distn.rvs(size=10)
 ```
 and to evaluate the probability mass function for some outcome, run
 ```python
 com_distn.pmf(99)
 com_distn.pmf(40)
 ```

## Estimating the parameters of the distribution
 To estmiate the parameters of the Conway Maxwell Binomial distribution given a sample, run the following lines
 ```python
 sample = com_distn.rvs(size=15)
 initial_params = [0.5, 1]
 cmb.estimateParams(m, sample, initial_params)
 ```
 To evaluate the negative log-likelihood of a sample, run
 ```python
 cmb.conwayMaxwellNegLogLike([p, nu], m, sample)
 ```
