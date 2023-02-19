import numpy as np
import tables
#from matplotlib.pyplot import figure, show
import scipy.linalg

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

def best_corr_vec(wvec, vocab, SU, n=10):
    """Returns the [n] words from [vocab] most similar to the given [wvec], where each word is represented
    as a row in [SU].  Similarity is computed using correlation."""
    wvec = wvec - np.mean(wvec)
    nwords = len(vocab)
    corrs = np.nan_to_num([np.corrcoef(wvec, SU[wi,:]-np.mean(SU[wi,:]))[1,0] for wi in range(nwords-1)])
    scorrs = np.argsort(corrs)
    words = list(reversed([(corrs[i],vocab[i]) for i in scorrs[-n:]]))
    return words

def get_word_prob():
    """Returns the probabilities of all the words in the mechanical turk video labels.
    """
    import constants as c
    import cPickle
    data = cPickle.load(open(c.datafile)) # Read in the words from the labels
    wordcount = dict()
    totalcount = 0
    for label in data:
        for word in label:
            totalcount += 1
            if word in wordcount:
                wordcount[word] += 1
            else:
                wordcount[word] = 1
    
    wordprob = dict([(word, float(wc)/totalcount) for word, wc in wordcount.items()])
    return wordprob

def best_prob_vec(wvec, vocab, space, wordprobs):
    """Orders the words by correlation with the given [wvec], but also weights the correlations by the prior
    probability of the word appearing in the mechanical turk video labels.
    """
    words = best_corr_vec(wvec, vocab, space, n=len(vocab)) ## get correlations for all words
    ## weight correlations by the prior probability of the word in the labels
    weightwords = []
    for wcorr,word in words:
        if word in wordprobs:
            weightwords.append((wordprobs[word]*wcorr, word))
    
    return sorted(weightwords, key=lambda ww: ww[0])

def find_best_words(vectors, vocab, wordspace, actual, display=True, num=15):
    cwords = []
    for si in range(len(vectors)):
        cw = best_corr_vec(vectors[si], vocab, wordspace, n=num)
        cwords.append(cw)
        if display:
            print ("Closest words to scene %d:" % si)
            print ([b[1] for b in cw])
            print ("Actual words:")
            print (actual[si])
            print ("")
    return cwords

def find_best_stims_for_word(wordvector, decstims, n):
    """Returns a list of the indexes of the [n] stimuli in [decstims] (should be decoded stimuli)
    that lie closest to the vector [wordvector], which should be taken from the same space as the
    stimuli.
    """
    scorrs = np.array([np.corrcoef(wordvector, ds)[0,1] for ds in decstims])
    scorrs[np.isnan(scorrs)] = -1
    return np.argsort(scorrs)[-n:][::-1]

def princomp(x, use_dgesvd=False):
    """Does principal components analysis on [x].
    Returns coefficients, scores and latent variable values.
    Translated from MATLAB princomp function.  Unlike the matlab princomp function, however, the
    rows of the returned value 'coeff' are the principal components, not the columns.
    """
    
    n,p = x.shape
    #cx = x-np.tile(x.mean(0), (n,1)) ## column-centered x
    cx = x-x.mean(0)
    r = np.min([n-1,p]) ## maximum possible rank of cx

    if use_dgesvd:
        from svd_dgesvd import svd_dgesvd
        U,sigma,coeff = svd_dgesvd(cx, full_matrices=False)
    else:
        U,sigma,coeff = np.linalg.svd(cx, full_matrices=False)
    
    sigma = np.diag(sigma)
    score = np.dot(cx, coeff.T)
    sigma = sigma/np.sqrt(n-1)
    
    latent = sigma**2

    return coeff, score, latent

def eigprincomp(x, npcs=None, norm=False, weights=None):
    """Does principal components analysis on [x].
    Returns coefficients (eigenvectors) and eigenvalues.
    If given, only the [npcs] greatest eigenvectors/values will be returned.
    If given, the covariance matrix will be computed using [weights] on the samples.
    """
    n,p = x.shape
    #cx = x-np.tile(x.mean(0), (n,1)) ## column-centered x
    cx = x-x.mean(0)
    r = np.min([n-1,p]) ## maximum possible rank of cx
    
    xcov = np.cov(cx.T)
    if norm:
        xcov /= n
    
    if npcs is not None:
        latent,coeff = scipy.linalg.eigh(xcov, eigvals=(p-npcs,p-1))
    else:
        latent,coeff = np.linalg.eigh(xcov)
    
    ## Transpose coeff, reverse its rows
    return coeff.T[::-1], latent[::-1]

def weighted_cov(x, weights=None):
    """If given [weights], the covariance will be computed using those weights on the samples.
    Otherwise the simple covariance will be returned.
    """
    if weights is None:
        return np.cov(x)
    else:
        w = weights/weights.sum() ## Normalize the weights
        dmx = (x.T-(w*x).sum(1)).T ## Subtract the WEIGHTED mean
        wfact = 1/(1-(w**2).sum()) ## Compute the weighting factor
        return wfact*np.dot(w*dmx, dmx.T.conj()) ## Take the weighted inner product

def test_weighted_cov():
    """Runs a test on the weighted_cov function, creating a dataset for which the covariance is known
    for two different populations, and weights are used to reproduce the individual covariances.
    """
    T = 1000 ## number of time points
    N = 100 ## A signals
    M = 100 ## B signals
    snr = 5 ## signal to noise ratio
    
    ## Create the two datasets
    siga = np.random.rand(T)
    noisea = np.random.rand(T, N)
    respa = (noisea.T+snr*siga).T

    sigb = np.random.rand(T)
    noiseb = np.random.rand(T, M)
    respb = (noiseb.T+snr*sigb).T

    ## Compute self-covariance matrixes
    cova = np.cov(respa)
    covb = np.cov(respb)

    ## Compute the full covariance matrix
    allresp = np.hstack([respa, respb])
    fullcov = np.cov(allresp)

    ## Make weights that will recover individual covariances
    wta = np.ones([N+M,])
    wta[N:] = 0

    wtb = np.ones([N+M,])
    wtb[:N] = 0

    recova = weighted_cov(allresp, wta)
    recovb = weighted_cov(allresp, wtb)
    
    return locals()

def fixPCs(orig, new):
    """Finds and fixes sign-flips in PCs by finding the coefficient with the greatest
    magnitude in the [orig] PCs, then negating the [new] PCs if that coefficient has
    a different sign.
    """
    flipped = []
    for o,n in zip(orig, new):
        maxind = np.abs(o).argmax()
        if o[maxind]*n[maxind]>0:
            ## Same sign, no need to flip
            flipped.append(n)
        else:
            ## Different sign, flip
            flipped.append(-n)
    
    return np.vstack(flipped)


def plot_model_comparison(corrs1, corrs2, name1, name2, thresh=0.35):
    fig = figure(figsize=(8,8))
    ax = fig.add_subplot(1,1,1)
    
    good1 = corrs1>thresh
    good2 = corrs2>thresh
    better1 = corrs1>corrs2
    #both = np.logical_and(good1, good2)
    neither = np.logical_not(np.logical_or(good1, good2))
    only1 = np.logical_and(good1, better1)
    only2 = np.logical_and(good2, np.logical_not(better1))
    
    ptalpha = 0.3
    ax.plot(corrs1[neither], corrs2[neither], 'ko', alpha=ptalpha)
    #ax.plot(corrs1[both], corrs2[both], 'go', alpha=ptalpha)
    ax.plot(corrs1[only1], corrs2[only1], 'ro', alpha=ptalpha)
    ax.plot(corrs1[only2], corrs2[only2], 'bo', alpha=ptalpha)
    
    lims = [-0.5, 1.0]
    
    ax.plot([thresh, thresh], [lims[0], thresh], 'r-')
    ax.plot([lims[0], thresh], [thresh,thresh], 'b-')
    
    ax.text(lims[0]+0.05, thresh, "$n=%d$"%np.sum(good2), horizontalalignment="left", verticalalignment="bottom")
    ax.text(thresh, lims[0]+0.05, "$n=%d$"%np.sum(good1), horizontalalignment="left", verticalalignment="bottom")
    
    ax.plot(lims, lims, '-', color="gray")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel(name1)
    ax.set_ylabel(name2)
    
    show()
    return fig

import matplotlib.colors
bwr = matplotlib.colors.LinearSegmentedColormap.from_list("bwr", ((0.0, 0.0, 1.0), (1.0, 1.0, 1.0), (1.0, 0.0, 0.0)))
bkr = matplotlib.colors.LinearSegmentedColormap.from_list("bkr", ((0.0, 0.0, 1.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)))
bgr = matplotlib.colors.LinearSegmentedColormap.from_list("bgr", ((0.0, 0.0, 1.0), (0.5, 0.5, 0.5), (1.0, 0.0, 0.0)))

def plot_model_comparison2(corrFile1, corrFile2, name1, name2, thresh=0.35):    
    fig = figure(figsize=(9,10))
    #ax = fig.add_subplot(3,1,[1,2], aspect="equal")
    ax = fig.add_axes([0.25, 0.4, 0.6, 0.5], aspect="equal")

    corrs1 = tables.openFile(corrFile1).root.semcorr.read()
    corrs2 = tables.openFile(corrFile2).root.semcorr.read()
    maxcorr = np.clip(np.vstack([corrs1, corrs2]).max(0), 0, thresh)/thresh
    corrdiff = (corrs1-corrs2) + 0.5
    colors = (bgr(corrdiff).T*maxcorr).T
    colors[:,3] = 1.0 ## Don't scale alpha
    
    ptalpha = 0.8
    ax.scatter(corrs1, corrs2, s=10, c=colors, alpha=ptalpha, edgecolors="none")
    lims = [-0.5, 1.0]
    
    ax.plot([thresh, thresh], [lims[0], thresh], color="gray")
    ax.plot([lims[0], thresh], [thresh,thresh], color="gray")

    good1 = corrs1>thresh
    good2 = corrs2>thresh
    ax.text(lims[0]+0.05, thresh, "$n=%d$"%np.sum(good2), horizontalalignment="left", verticalalignment="bottom")
    ax.text(thresh, lims[0]+0.05, "$n=%d$"%np.sum(good1), horizontalalignment="left", verticalalignment="bottom")
    
    ax.plot(lims, lims, '-', color="gray")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel(name1+" model")
    ax.set_ylabel(name2+" model")

    fig.canvas.draw()
    show()
    ## Add over-under comparison
    #ax_left = ax.get_window_extent()._bbox.x0
    #ax_right = ax.get_window_extent()._bbox.x1
    #ax_width = ax_right-ax_left
    #print ax_left, ax_right
    #ax2 = fig.add_axes([ax_left, 0.1, ax_width, 0.2])
    ax2 = fig.add_axes([0.25, 0.1, 0.6, 0.25])#, sharex=ax)
    #ax2 = fig.add_subplot(3, 1, 3)
    #plot_model_overunder_comparison(corrs1, corrs2, name1, name2, thresh=thresh, ax=ax2)
    plot_model_histogram_comparison(corrs1, corrs2, name1, name2, thresh=thresh, ax=ax2)

    fig.suptitle("Model comparison: %s vs. %s"%(name1, name2))
    show()
    return fig


def plot_model_overunder_comparison(corrs1, corrs2, name1, name2, thresh=0.35, ax=None):
    """Plots over-under difference between two models.
    """
    if ax is None:
        fig = figure(figsize=(8,8))
        ax = fig.add_subplot(1,1,1)

    maxcorr = max(corrs1.max(), corrs2.max())
    vals = np.linspace(0, maxcorr, 500)
    overunder = lambda c: np.array([np.sum(c>v)-np.sum(c<-v) for v in vals])

    ou1 = overunder(corrs1)
    ou2 = overunder(corrs2)

    oud = ou2-ou1

    ax.fill_between(vals, 0, np.clip(oud, 0, 1e9), facecolor="blue")
    ax.fill_between(vals, 0, np.clip(oud, -1e9, 0), facecolor="red")

    yl = np.max(np.abs(np.array(ax.get_ylim())))
    ax.plot([thresh, thresh], [-yl, yl], '-', color="gray")
    ax.set_ylim(-yl, yl)
    ax.set_xlim(0, maxcorr)
    ax.set_xlabel("Voxel correlation")
    ax.set_ylabel("%s better           %s better"%(name1, name2))

    show()
    return ax

def plot_model_histogram_comparison(corrs1, corrs2, name1, name2, thresh=0.35, ax=None):
    """Plots over-under difference between two models.
    """
    if ax is None:
        fig = figure(figsize=(8,8))
        ax = fig.add_subplot(1,1,1)
    
    maxcorr = max(corrs1.max(), corrs2.max())
    nbins = 100
    hist1 = np.histogram(corrs1, nbins, range=(-1,1))
    hist2 = np.histogram(corrs2, nbins, range=(-1,1))

    ouhist1 = hist1[0][nbins/2:]-hist1[0][:nbins/2][::-1]
    ouhist2 = hist2[0][nbins/2:]-hist2[0][:nbins/2][::-1]

    oud = ouhist2-ouhist1
    bwidth = 2.0/nbins
    barlefts = hist1[1][nbins/2:-1]

    #ax.fill_between(vals, 0, np.clip(oud, 0, 1e9), facecolor="blue")
    #ax.fill_between(vals, 0, np.clip(oud, -1e9, 0), facecolor="red")

    ax.bar(barlefts, np.clip(oud, 0, 1e9), bwidth, facecolor="blue")
    ax.bar(barlefts, np.clip(oud, -1e9, 0), bwidth, facecolor="red")

    yl = np.max(np.abs(np.array(ax.get_ylim())))
    ax.plot([thresh, thresh], [-yl, yl], '-', color="gray")
    ax.set_ylim(-yl, yl)
    ax.set_xlim(0, maxcorr)
    ax.set_xlabel("Voxel correlation")
    ax.set_ylabel("%s better           %s better"%(name1, name2))

    show()
    return ax


def plot_model_comparison_rois(corrs1, corrs2, name1, name2, roivoxels, roinames, thresh=0.35):
    """Plots model correlation comparisons per ROI.
    """
    fig = figure()
    ptalpha = 0.3
    
    for ri in range(len(roinames)):
        ax = fig.add_subplot(4, 4, ri+1)
        ax.plot(corrs1[roivoxels[ri]], corrs2[roivoxels[ri]], 'bo', alpha=ptalpha)
        lims = [-0.3, 1.0]
        ax.plot(lims, lims, '-', color="gray")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_title(roinames[ri])
    
    show()
    return fig

def save_table_file(filename, filedict):
    """Saves the variables in [filedict] in a hdf5 table file at [filename].
    """
    hf = tables.openFile(filename, mode="w", title="save_file")
    for vname, var in filedict.items():
        hf.createArray("/", vname, var)
    hf.close()