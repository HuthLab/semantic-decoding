import numpy as np
#import scipy.stats
import random
import sys

def zscore(mat, return_unzvals=False):
    """Z-scores the rows of [mat] by subtracting off the mean and dividing
    by the standard deviation.
    If [return_unzvals] is True, a matrix will be returned that can be used
    to return the z-scored values to their original state.
    """
    zmat = np.empty(mat.shape, mat.dtype)
    unzvals = np.zeros((zmat.shape[0], 2), mat.dtype)
    for ri in range(mat.shape[0]):
        unzvals[ri,0] = np.std(mat[ri,:])
        unzvals[ri,1] = np.mean(mat[ri,:])
        zmat[ri,:] = (mat[ri,:]-unzvals[ri,1]) / (1e-10+unzvals[ri,0])
    
    if return_unzvals:
        return zmat, unzvals
    
    return zmat

def center(mat, return_uncvals=False):
    """Centers the rows of [mat] by subtracting off the mean, but doesn't 
    divide by the SD.
    Can be undone like zscore.
    """
    cmat = np.empty(mat.shape)
    uncvals = np.ones((mat.shape[0], 2))
    for ri in range(mat.shape[0]):
        uncvals[ri,1] = np.mean(mat[ri,:])
        cmat[ri,:] = mat[ri,:]-uncvals[ri,1]
    
    if return_uncvals:
        return cmat, uncvals
    
    return cmat

def unzscore(mat, unzvals):
    """Un-Z-scores the rows of [mat] by multiplying by unzvals[:,0] (the standard deviations)
    and then adding unzvals[:,1] (the row means).
    """
    unzmat = np.empty(mat.shape)
    for ri in range(mat.shape[0]):
        unzmat[ri,:] = mat[ri,:]*(1e-10+unzvals[ri,0])+unzvals[ri,1]
    return unzmat

def gaussianize(vec):
    """Uses a look-up table to force the values in [vec] to be gaussian."""
    ranks = np.argsort(np.argsort(vec))
    cranks = (ranks+1).astype(float)/(ranks.max()+2)
    vals = scipy.stats.norm.isf(1-cranks)
    zvals = vals/vals.std()
    return zvals

def gaussianize_mat(mat):
    """Gaussianizes each column of [mat]."""
    gmat = np.empty(mat.shape)
    for ri in range(mat.shape[1]):
        gmat[:,ri] = gaussianize(mat[:,ri])
    return gmat

def make_delayed(stim, delays, circpad=False):
    """Creates non-interpolated concatenated delayed versions of [stim] with the given [delays] 
    (in samples).
    
    If [circpad], instead of being padded with zeros, [stim] will be circularly shifted.
    """
    nt,ndim = stim.shape
    dstims = []
    for di,d in enumerate(delays):
        dstim = np.zeros((nt, ndim))
        if d<0: ## negative delay
            dstim[:d,:] = stim[-d:,:]
            if circpad:
                dstim[d:,:] = stim[:-d,:]
        elif d>0:
            dstim[d:,:] = stim[:-d,:]
            if circpad:
                dstim[:d,:] = stim[-d:,:]
        else: ## d==0
            dstim = stim.copy()
        dstims.append(dstim)
    return np.hstack(dstims)

def mult_diag(d, mtx, left=True):
    """Multiply a full matrix by a diagonal matrix.
    This function should always be faster than dot.
    Input:
      d -- 1D (N,) array (contains the diagonal elements)
      mtx -- 2D (N,N) array
    Output:
      mult_diag(d, mts, left=True) == dot(diag(d), mtx)
      mult_diag(d, mts, left=False) == dot(mtx, diag(d))
    
    By Pietro Berkes
    From http://mail.scipy.org/pipermail/numpy-discussion/2007-March/026807.html
    """
    if left:
        return (d*mtx.T).T
    else:
        return d*mtx

import time
import logging
def counter(iterable, countevery=100, total=None, logger=logging.getLogger("counter")):
    """Logs a status and timing update to [logger] every [countevery] draws from [iterable].
    If [total] is given, log messages will include the estimated time remaining.
    """
    start_time = time.time()

    ## Check if the iterable has a __len__ function, use it if no total length is supplied
    if total is None:
        if hasattr(iterable, "__len__"):
            total = len(iterable)
    
    for count, thing in enumerate(iterable):
        yield thing
        
        if not count%countevery:
            current_time = time.time()
            rate = float(count+1)/(current_time-start_time)

            if rate>1: ## more than 1 item/second
                ratestr = "%0.2f items/second"%rate
            else: ## less than 1 item/second
                ratestr = "%0.2f seconds/item"%(rate**-1)
            
            if total is not None:
                remitems = total-(count+1)
                remtime = remitems/rate
                timestr = ", %s remaining" % time.strftime('%H:%M:%S', time.gmtime(remtime))
                itemstr = "%d/%d"%(count+1, total)
            else:
                timestr = ""
                itemstr = "%d"%(count+1)

            formatted_str = "%s items complete (%s%s)"%(itemstr,ratestr,timestr)
            if logger is None:
                print(formatted_str)
            else:
                logger.info(formatted_str)