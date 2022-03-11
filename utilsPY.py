import numpy as np
from lmoments3 import distr
from scipy.stats import gamma
import scipy.stats
import logging
from scipy.optimize import minimize
# ------------------------------------------------------------------------------
#set up a basic, global _logger
def get_logger(name, level):
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d  %H:%M:%S",
    )
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger

# ------------------------------------------------------------------------------
def scale_values(
        values: np.ndarray,
        scale: int,
        periodicity: str,
):
    # we expect to operate upon a 1-D array, so if we've been passed a 2-D array
    # then we flatten it, otherwise raise an error
    shape = values.shape
    if len(shape) == 2:
        values = values.flatten()
    elif len(shape) != 1:
        message = "Invalid shape of input array: {shape}".format(shape=shape) + \
                  " -- only 1-D and 2-D arrays are supported"
        get_logger.error(message)
        raise ValueError(message)

    # if we're passed all missing values then we can't compute
    # anything, so we return the same array of missing values
    if (np.ma.is_masked(values) and values.mask.all()) or np.all(np.isnan(values)):
        return values

    # clip any negative values to zero
    if np.amin(values) < 0.0:
        get_logger.warn("Input contains negative values -- all negatives clipped to zero")
        values = np.clip(values, a_min=0.0, a_max=None)

    # get a sliding sums array, with each time step's value scaled
    # by the specified number of time steps
    scaled_values = sum_to_scale(values, scale)

    # reshape precipitation values to (years, 12) for monthly,
    # or to (years, 366) for daily
    if periodicity == 'daily':

        scaled_values = scaled_values.reshape((int(scaled_values.shape[0] / 365), 365))

    elif periodicity == 'monthly':

        scaled_values = scaled_values.reshape((int(scaled_values.shape[0] / 12), 12))

    else:

        raise ValueError("Invalid periodicity argument: %s" % periodicity)

    return scaled_values

# ------------------------------------------------------------------------------
def sum_to_scale(
        values: np.ndarray,
        scale: int,
) -> np.ndarray:
    """
    Compute the moving average according to the given scale
    values:A one-dimensional array that needs to be convolved
    scale:Scale integer, it can be 30,60,90,120...
    """

    # don't bother if the number of values to sum is 1
    if scale == 1:
        return values

    # get the valid sliding summations with 1D convolution
    sliding_sums = np.convolve(values, np.ones(scale), mode="valid")

    # pad the first (n - 1) elements of the array with NaN values
    return np.hstack(([np.NaN] * (scale - 1), sliding_sums))/scale

# ------------------------------------------------------------------------------
def fun(al):
    v=lambda x:-np.sum(scipy.stats.gamma.logpdf(al,a=x[0],scale=x[1]))
    return v
def fit_gamma_para(
        x: np.ndarray,
        p0='TRUE',
        mass='TRUE',
        fix='TRUE',
):
    #The maximum likelihood method is used to estimate the parameters of the Gamma distribution
    #x:Value that has been convolved and reshape
    # and more distributions will be explored in the future.
    # p0，mass，and fix:The default Settings are used and more will be explored in the future
    b=x

    #Initial parameters
    result_alpa = np.zeros((365)) - 999
    result_beta = np.zeros((365)) - 999

    #Maximum likelihood fitting parameter
    f_alpa = np.zeros((365)) - 999
    f_beta = np.zeros((365)) - 999

    #The probability of zero
    f_P0 = np.zeros((365)) - 999

    for i in range(0, 365):
        now = b[:, i]
        now = now[now == now]

        #If there is no data, the value is null
        if now.shape[0] == 0:
            result_alpa[i] = np.nan
            result_beta[i] = np.nan
            f_alpa[i] = np.nan
            f_beta[i] = np.nan
            f_P0[i] = np.nan
            continue

        # If all values are the same, the value is null
        elif np.all(now == now[0]):
            result_alpa[i] = np.nan
            result_beta[i] = np.nan
            f_alpa[i] = np.nan
            f_beta[i] = np.nan
            f_P0[i] = np.nan
            continue

        #Calculate the probability of zero
        npo = now[now == 0].shape[0]
        nn = now.shape[0]
        if mass:
            est = npo / (nn + 1)
        else:
            est = npo / nn

        #If 0 exists and the non-zero values are greater than three, the non-zero values are picked out for likelihood estimation.
        #Add a value close to zero at the end to prevent gaps between the data
        # Initial parameters were estimated by L-moments
        if est > 0 and (nn - npo > 3):
            now = now[now > 0]
            now = np.append(now, np.min(now) * 0.01)
            paras = distr.gam.lmom_fit(now)
            result_alpa[i] = paras['a']
            result_beta[i] = paras['scale']
            aa = now
            ss = np.asarray((paras['a'],paras['scale']))
            aal = minimize(fun(aa), x0=ss, method='Nelder-Mead')
            kkk = aal.x
            f_P0[i] = est
            f_alpa[i] = kkk[0]
            f_beta[i] = kkk[1]

        #If there is no value of 0, the parameter estimation is performed directly
        #Initial parameters were estimated by L-moments
        elif est == 0:
            now = now[now > 0]
            paras = distr.gam.lmom_fit(now)
            result_alpa[i] = paras['a']
            result_beta[i] = paras['scale']
            aa = now
            ss = np.asarray((paras['a'],paras['scale']))
            aal = minimize(fun(aa), x0=ss, method='Nelder-Mead')
            kkk = aal.x
            f_P0[i] = est
            f_alpa[i] = kkk[0]
            f_beta[i] = kkk[1]

        #If there are zero values and the number of non-zero values is less than three, the Moments method is used for initial parameter estimation
        elif est > 0 and (nn - npo <= 3) and (nn - npo >= 1):
            now = now[now > 0]
            now = np.append(now, np.min(now) * 0.01)
            n = now.shape[0]
            m = np.mean(now)
            v = np.var(now)
            shape = m ** 2 / v
            rate = m / v
            result_alpa[i] = shape
            result_beta[i] = rate
            aa = now
            ss = np.asarray((shape,rate))
            aal = minimize(fun(aa), x0=ss, method='Nelder-Mead')
            kkk = aal.x
            f_P0[i] = est
            f_alpa[i] = kkk[0]
            f_beta[i] = kkk[1]

        #If all values are 0, the argument is null
        elif est > 0 and (nn == npo):
            result_alpa[i] = np.nan
            result_beta[i] = np.nan
            f_alpa[i] = np.nan
            f_beta[i] = np.nan
            f_P0[i] = np.nan

#Returns the paras and probability of a zero value
    return f_alpa,f_beta,f_P0

# ------------------------------------------------------------------------------
def caculate_gamma(
        x: np.ndarray,
        alpha: np.ndarray,
        belta: np.ndarray,
        p0:np.ndarray,
):
    """
        Fit values to a gamma distribution and transform the values to corresponding
        normalized sigmas.

        :param x: 2-D array of values, with each row typically representing a year
                       containing twelve columns representing the respective calendar
                       months, or 366 days per column as if all years were leap years
        :param alphas: pre-computed gamma fitting parameters
        :param belta: pre-computed gamma fitting parameters
        :param p0: pre-computed gamma fitting parameters
        :return: SPI
        """
    alldata=np.zeros(x.shape)-999

    for i in range(0,365):
        nowd=x[:,i]
        nowd = nowd[nowd == nowd]
        nnn = nowd.shape[0]
        nnnp0 = nowd[nowd == 0].shape[0]
        data = gamma.cdf(nowd, a=alpha[i], scale=belta[i])
        spi = p0[i] + (1 - p0[i]) * data
        spi[nowd == 0] = (nnnp0 + 1) / (2 * (nnn + 1))
        if nnn == nnnp0:
            spi[nowd == 0] = np.nan
        alldata[x.shape[0]- nnn:, i] = spi

    alldata[alldata == -999] = np.nan
    SPI = scipy.stats.norm.ppf(alldata)
    return SPI
def dgev(x, loc = 0, scale = 1, shape = 0, log = False):
    x = (x - loc)/scale
    if shape == 0:
        d = np.log(1/scale) - x - np.exp(-x)
    else:
        nn = x.shape[0]
        xx = 1 + shape*x
        xxpos = xx[xx>0]
        scale = np.zeros((nn))[xx>0]+scale
        d = np.zeros((nn))
        d[xx>0] = np.log(1/scale) - xxpos**(-1/shape) -(1/shape + 1)*np.log(xxpos)
        d[xx<=0] = -np.inf
    if not log:
        d = np.exp(d)
    return d

def funspei(al):
    v = lambda x: -np.sum(dgev(x=al, shape=x[0], loc=x[1], scale=x[2], log=True))
    return v

def scale_values_spei(
        values: np.ndarray,
        scale: int,
        periodicity: str,
):
    # we expect to operate upon a 1-D array, so if we've been passed a 2-D array
    # then we flatten it, otherwise raise an error
    shape = values.shape
    if len(shape) == 2:
        values = values.flatten()
    elif len(shape) != 1:
        message = "Invalid shape of input array: {shape}".format(shape=shape) + \
                  " -- only 1-D and 2-D arrays are supported"
        get_logger.error(message)
        raise ValueError(message)

    # if we're passed all missing values then we can't compute
    # anything, so we return the same array of missing values
    if (np.ma.is_masked(values) and values.mask.all()) or np.all(np.isnan(values)):
        return values

    # clip any negative values to zero

    # get a sliding sums array, with each time step's value scaled
    # by the specified number of time steps
    scaled_values = sum_to_scale(values, scale)

    # reshape precipitation values to (years, 12) for monthly,
    # or to (years, 366) for daily
    if periodicity == 'daily':

        scaled_values = scaled_values.reshape((int(scaled_values.shape[0] / 365), 365))

    elif periodicity == 'monthly':

        scaled_values = scaled_values.reshape((int(scaled_values.shape[0] / 12), 12))

    else:

        raise ValueError("Invalid periodicity argument: %s" % periodicity)

    return scaled_values

def fit_gev_para(
    x:np.ndarray,
):
    #Initial parameters
    result_alpa = np.zeros((365)) - 999
    result_loca = np.zeros((365)) - 999
    result_beta = np.zeros((365)) - 999

    #Maximum likelihood fitting parameter
    f_alpa = np.zeros((365)) - 999
    f_loca = np.zeros((365)) - 999
    f_beta = np.zeros((365)) - 999

    for i in range(0, 365):
        now = x[:, i]
        now = now[now == now]

        #If there is no data, the value is null
        if now.shape[0] == 0:
            result_alpa[i] = np.nan
            result_loca[i]=np.nan
            result_beta[i] = np.nan
            f_alpa[i] = np.nan
            f_loca[i] = np.nan
            f_beta[i] = np.nan
            continue

        # If all values are the same, the value is null
        elif np.all(now == now[0]):
            result_alpa[i] = np.nan
            result_loca[i] = np.nan
            result_beta[i] = np.nan
            f_alpa[i] = np.nan
            f_loca[i] = np.nan
            f_beta[i] = np.nan
            continue

        else:
            try:
                paras = distr.gev.lmom_fit(now)
            except (ValueError):
                f_alpa[i] = np.nan
                f_loca[i] = np.nan
                f_beta[i] = np.nan
                continue
            result_alpa[i] = -paras['c']
            result_loca[i]=paras['loc']
            result_beta[i] = paras['scale']
            aa = now
            ss = np.asarray((-paras['c'],paras['loc'],paras['scale']))
            aal = minimize(funspei(aa), ss, method='Nelder-Mead')
            kkk = aal.x
            f_alpa[i] = kkk[0]
            f_loca[i] = kkk[1]
            f_beta[i] = kkk[2]
            if f_alpa[i]==0 and f_loca[i]==0 and f_beta[i]==0:
                f_alpa[i] = -paras['c']
                f_loca[i] = paras['loc']
                f_beta[i] = paras['scale']
    return f_alpa,f_loca,f_beta
def pgev(q, loc = 0, scale = 1, shape = 0, lower = True):
    q = (q - loc)/scale
    if shape == 0:
        p = np.exp(-np.exp(-q))
    else:
        qq=1+shape * q
        qq[qq<0]=0
        p = np.exp(-qq**(-1/shape))
    if not lower:
        p = 1 - p
    return p
def caculate_SPEI_gev(
        x: np.ndarray,
        alpha: np.ndarray,
        loc:np.ndarray,
        belta: np.ndarray,
):
    alldata=np.zeros(x.shape)-999
    for i in range(0, 365):
        nowd = x[:, i]
        nowdf = nowd[nowd == nowd]
        if alpha[i] == alpha[i] and loc[i] == loc[i] and belta[i] == belta[i]:
            data = pgev(nowdf, loc=loc[i], scale=belta[i], shape=alpha[i])
            alldata[int(x.shape[0] - nowdf.shape[0]):, i] = list(data)
        else:
            alldata[int(x.shape[0] - nowdf.shape[0]):, i] = np.nan
    alldata[alldata == -999] = np.nan
    SS = scipy.stats.norm.ppf(alldata)
    return SS