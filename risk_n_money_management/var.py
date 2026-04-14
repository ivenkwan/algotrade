# var.py

from datetime import datetime as dt
import os
import sys
sys.path.append(os.path.join('..', 'pricing'))

import numpy as np
from scipy.stats import norm

from alpha_vantage import AlphaVantage


def var_cov_var(P, c, mu, sigma):
    """
    Variance-Covariance calculation of daily Value-at-Risk
    using confidence level c, with mean of returns mu
    and standard deviation of returns sigma, on a portfolio
    of value P.

    Parameters
    ----------
    P : `float`
        The total value of the portfolio
    c : `float`
        The confidence level (e.g. 0.99 for 99%)
    mu : `float`
        The mean of the returns
    sigma : `float`
        The standard deviation of the returns

    Returns
    -------
    `float`
        The calculated Value-at-Risk (VaR)
    """
    alpha = norm.ppf(1-c, mu, sigma)
    return P - P*(alpha + 1)

if __name__ == "__main__":
    # Create an AlphaVantage API instance
    av = AlphaVantage()
    
    # Download the Citi Group OHLCV data from 1/1/2010 to 1/1/2014
    start_date = dt(2010, 1, 1)
    end_date = dt(2014, 1, 1)
    citi = av.get_daily_historic_data('C', start_date, end_date)

    # Calculate the percentage change
    citi["rets"] = citi["Adj Close"].pct_change()

    P = 1e6   # 1,000,000 USD
    c = 0.99  # 99% confidence interval
    mu = np.mean(citi["rets"])
    sigma = np.std(citi["rets"])

    var = var_cov_var(P, c, mu, sigma)
    print("Value-at-Risk: $%0.2f" % var)
