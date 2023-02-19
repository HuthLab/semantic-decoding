import numpy as np
import logging

logger = logging.getLogger("text.regression.interpdata")

def interpdata(data, oldtime, newtime):
    """Interpolates the columns of [data] to find the values at [newtime], given that the current
    values are at [oldtime].  [oldtime] must have the same number of elements as [data] has rows.
    """
    ## Check input sizes ##
    if not len(oldtime) == data.shape[0]:
        raise IndexError("oldtime must have same number of elements as data has rows.")
    
    ## Set up matrix to hold output ##
    newdata = np.empty((len(newtime), data.shape[1]))
    
    ## Interpolate each column of data ##
    for ci in range(data.shape[1]):
        if (ci%100) == 0:
            logger.info("Interpolating column %d/%d.." % (ci+1, data.shape[1]))
        
        newdata[:,ci] = np.interp(newtime, oldtime, data[:,ci])
    
    ## Return interpolated data ##
    return newdata

def sincinterp1D(data, oldtime, newtime, cutoff_mult=1.0, window=1):
    """Interpolates the one-dimensional signal [data] at the times given by [newtime], assuming
    that each sample in [data] was collected at the corresponding time in [oldtime]. Clearly,
    [oldtime] and [data] must have the same length, but [newtime] can have any length.
    
    This function will assume that the time points in [newtime] are evenly spaced and will use
    that frequency multipled by [cutoff_mult] as the cutoff frequency of the sinc filter.
    
    The sinc function will be computed with [window] lobes.  With [window]=1, this will
    effectively compute the Lanczos filter.
    
    This is a very simplistic filtering algorithm, so will take O(N*M) time, where N is the
    length of [oldtime] and M is the length of [newtime].
    
    This filter is non-causal.
    """
    ## Find the cutoff frequency ##
    cutoff = 1/np.mean(np.diff(newtime)) * cutoff_mult
    print("Doing sinc interpolation with cutoff=%0.3f and %d lobes." % (cutoff, window))
    
    ## Construct new signal ##
    newdata = np.zeros((len(newtime),1))
    for ndi in range(len(newtime)):
        for di in range(len(oldtime)):
            newdata[ndi] += sincfun(cutoff, newtime[ndi]-oldtime[di], window) * data[di]
    return newdata

def sincinterp2D(data, oldtime, newtime, cutoff_mult=1.0, window=1, causal=False, renorm=True):
    """Interpolates the columns of [data], assuming that the i'th row of data corresponds to
    oldtime(i).  A new matrix with the same number of columns and a number of rows given
    by the length of [newtime] is returned.  If [causal], only past time points will be used
    to computed the present value, and future time points will be ignored.
    
    The time points in [newtime] are assumed to be evenly spaced, and their frequency will
    be used to calculate the low-pass cutoff of the sinc interpolation filter.
    
    [window] lobes of the sinc function will be used.  [window] should be an integer.
    """
    ## Find the cutoff frequency ##
    cutoff = 1/np.mean(np.diff(newtime)) * cutoff_mult
    print("Doing sinc interpolation with cutoff=%0.3f and %d lobes." % (cutoff, window))
    
    ## Construct new signal ##
    # newdata = np.zeros((len(newtime), data.shape[1]))
    # for ndi in range(len(newtime)):
    #         for di in range(len(oldtime)):
    #             newdata[ndi,:] += sincfun(cutoff, newtime[ndi]-oldtime[di], window, causal) * data[di,:]
    
    ## Build up sinc matrix ##
    sincmat = np.zeros((len(newtime), len(oldtime)))
    for ndi in range(len(newtime)):
        sincmat[ndi,:] = sincfun(cutoff, newtime[ndi]-oldtime, window, causal, renorm)
    
    ## Construct new signal by multiplying the sinc matrix by the data ##
    newdata = np.dot(sincmat, data)

    return newdata

def lanczosinterp2D(data, oldtime, newtime, window=3, cutoff_mult=1.0, rectify=False):
    """Interpolates the columns of [data], assuming that the i'th row of data corresponds to
    oldtime(i). A new matrix with the same number of columns and a number of rows given
    by the length of [newtime] is returned.
    
    The time points in [newtime] are assumed to be evenly spaced, and their frequency will
    be used to calculate the low-pass cutoff of the interpolation filter.
    
    [window] lobes of the sinc function will be used. [window] should be an integer.
    """
    ## Find the cutoff frequency ##
    cutoff = 1/np.mean(np.diff(newtime)) * cutoff_mult
    # print "Doing lanczos interpolation with cutoff=%0.3f and %d lobes." % (cutoff, window)
    
    ## Build up sinc matrix ##
    sincmat = np.zeros((len(newtime), len(oldtime)))
    for ndi in range(len(newtime)):
        sincmat[ndi,:] = lanczosfun(cutoff, newtime[ndi]-oldtime, window)
    
    if rectify:
        newdata = np.hstack([np.dot(sincmat, np.clip(data, -np.inf, 0)), 
                            np.dot(sincmat, np.clip(data, 0, np.inf))])
    else:
        ## Construct new signal by multiplying the sinc matrix by the data ##
        newdata = np.dot(sincmat, data)

    return newdata

def sincupinterp2D(data, oldtime, newtimes, cutoff, window=1):
    """Uses sinc interpolation to upsample the columns of [data], assuming that the i'th
    row of data comes from oldtime[i].  A new matrix with the same number of columns
    and a number of rows given by the length of [newtime] is returned.
    The times points in [oldtime] are assumed to be evenly spaced, and their frequency
    will be used to calculate the low-pass cutoff of the sinc interpolation filter.
    [window] lobes of the sinc function will be used.  [window] should be an integer.
    Setting [window] to 1 yields a Lanczos filter.
    """
    #cutoff = 1/np.mean(np.diff(oldtime))
    print("Doing sinc interpolation with cutoff=%0.3f and %d lobes."%(cutoff, window))
    
    sincmat = np.zeros((len(newtimes), len(oldtime)))
    for ndi in range(len(newtimes)):
        sincmat[ndi,:] = sincfun(cutoff, newtimes[ndi]-oldtime, window, False)

    newdata = np.dot(sincmat, data)
    return newdata

def sincfun(B, t, window=np.inf, causal=False, renorm=True):
    """Compute the sinc function with some cutoff frequency [B] at some time [t].
    [t] can be a scalar or any shaped numpy array.
    If given a [window], only the lowest-order [window] lobes of the sinc function
    will be non-zero.
    If [causal], only past values (i.e. t<0) will have non-zero weights.
    """
    val = 2*B*np.sin(2*np.pi*B*t)/(2*np.pi*B*t+1e-20)
    if t.shape:
        val[np.abs(t)>window/(2*B)] = 0
        if causal:
            val[t<0] = 0
        if not np.sum(val)==0.0 and renorm:
            val = val/np.sum(val)
    elif np.abs(t)>window/(2*B):
        val = 0
        if causal and t<0:
            val = 0
    return val

def lanczosfun(cutoff, t, window=3):
    """Compute the lanczos function with some cutoff frequency [B] at some time [t].
    [t] can be a scalar or any shaped numpy array.
    If given a [window], only the lowest-order [window] lobes of the sinc function
    will be non-zero.
    """
    t = t * cutoff
    val = window * np.sin(np.pi*t) * np.sin(np.pi*t/window) / (np.pi**2 * t**2)
    val[t==0] = 1.0
    val[np.abs(t)>window] = 0.0
    return val# / (val.sum() + 1e-10)

def expinterp2D(data, oldtime, newtime, theta):
    intmat = np.zeros((len(newtime), len(oldtime)))
    for ndi in range(len(newtime)):
        intmat[ndi,:] = expfun(theta, newtime[ndi]-oldtime)
    
    ## Construct new signal by multiplying the sinc matrix by the data ##
    newdata = np.dot(intmat, data)
    return newdata

def expfun(theta, t):
    """Computes an exponential weighting function for interpolation.
    """
    val = np.exp(-t*theta)
    val[t<0] = 0.0
    if not np.sum(val)==0.0:
        val = val/np.sum(val)
    return val

def gabor_xfm(data, oldtimes, newtimes, freqs, sigma):
    sinvals = np.vstack([np.sin(oldtimes*f*2*np.pi) for f in freqs])
    cosvals = np.vstack([np.cos(oldtimes*f*2*np.pi) for f in freqs])
    outvals = np.zeros((len(newtimes), len(freqs)), dtype=np.complex128)
    for ti,t in enumerate(newtimes):
        ## Build gaussian function
        gaussvals = np.exp(-0.5*(oldtimes-t)**2/(2*sigma**2))*data
        ## Take product with sin/cos vals
        sprod = np.dot(sinvals, gaussvals)
        cprod = np.dot(cosvals, gaussvals)
        ## Store the output
        outvals[ti,:] = cprod + 1j*sprod

    return outvals

def gabor_xfm2D(ddata, oldtimes, newtimes, freqs, sigma):
    return np.vstack([gabor_xfm(d, oldtimes, newtimes, freqs, sigma).T for d in ddata])

def test_interp(**kwargs):
    """Tests sincinterp2D passing it the given [kwargs] and interpolating known signals 
    between the two time domains.
    """
    oldtime = np.linspace(0, 10, 100)
    newtime = np.linspace(0, 10, 49)
    data = np.zeros((4, 100))
    ## The first row has a single nonzero value
    data[0,50] = 1.0
    ## The second row has a few nonzero values in a row
    data[1,45:55] = 1.0
    ## The third row has a few nonzero values separated by zeros
    data[2,40:45] = 1.0
    data[2,55:60] = 1.0
    ## The fourth row has different values
    data[3,40:45] = 1.0
    data[3,55:60] = 2.0
    
    ## Interpolate the data
    interpdata = sincinterp2D(data.T, oldtime, newtime, **kwargs).T
    
    ## Plot the results
    from matplotlib.pyplot import figure, show
    fig = figure()
    for d in range(4):
        ax = fig.add_subplot(4,1,d+1)
        ax.plot(newtime, interpdata[d,:], 'go-')
        ax.plot(oldtime, data[d,:], 'bo-')
        
        #ax.tight()
    show()
    return newtime, interpdata