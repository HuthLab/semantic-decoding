#import scipy
from functools import reduce
import numpy as np
import logging
from utils_ridge.utils import mult_diag, counter
import random
import itertools as itools

zs = lambda v: (v-v.mean(0))/v.std(0) ## z-score function

def ridge(stim, resp, alpha, singcutoff=1e-10, normalpha=False):
    """Uses ridge regression to find a linear transformation of [stim] that approximates
    [resp]. The regularization parameter is [alpha].
    Parameters
    ----------
    stim : array_like, shape (T, N)
        Stimuli with T time points and N features.
    resp : array_like, shape (T, M)
        Responses with T time points and M separate responses.
    alpha : float or array_like, shape (M,)
        Regularization parameter. Can be given as a single value (which is applied to
        all M responses) or separate values for each response.
    normalpha : boolean
        Whether ridge parameters should be normalized by the largest singular value of stim. Good for
        comparing models with different numbers of parameters.
    Returns
    -------
    wt : array_like, shape (N, M)
        Linear regression weights.
    """
    try:
        U,S,Vh = np.linalg.svd(stim, full_matrices=False)
    except np.linalg.LinAlgError:
        from text.regression.svd_dgesvd import svd_dgesvd
        U,S,Vh = svd_dgesvd(stim, full_matrices=False)

    UR = np.dot(U.T, np.nan_to_num(resp))
    
    # Expand alpha to a collection if it's just a single value
    if isinstance(alpha, (float,int)):
        alpha = np.ones(resp.shape[1]) * alpha
    
    # Normalize alpha by the LSV norm
    norm = S[0]
    if normalpha:
        nalphas = alpha * norm
    else:
        nalphas = alpha

    # Compute weights for each alpha
    ualphas = np.unique(nalphas)
    wt = np.zeros((stim.shape[1], resp.shape[1]))
    for ua in ualphas:
        selvox = np.nonzero(nalphas==ua)[0]
        #awt = reduce(np.dot, [Vh.T, np.diag(S/(S**2+ua**2)), UR[:,selvox]])
        awt = Vh.T.dot(np.diag(S/(S**2+ua**2))).dot(UR[:,selvox])
        wt[:,selvox] = awt

    return wt

def ridge_corr(Rstim, Pstim, Rresp, Presp, alphas, normalpha=False, dtype=np.single, corrmin=0.2,
               singcutoff=1e-10, use_corr=True, logger=logging.getLogger("ridge_corr")):
    """Uses ridge regression to find a linear transformation of [Rstim] that approximates [Rresp].
    Then tests by comparing the transformation of [Pstim] to [Presp]. This procedure is repeated
    for each regularization parameter alpha in [alphas]. The correlation between each prediction and
    each response for each alpha is returned. Note that the regression weights are NOT returned.
    Parameters
    ----------
    Rstim : array_like, shape (TR, N)
        Training stimuli with TR time points and N features. Each feature should be Z-scored across time.
    Pstim : array_like, shape (TP, N)
        Test stimuli with TP time points and N features. Each feature should be Z-scored across time.
    Rresp : array_like, shape (TR, M)
        Training responses with TR time points and M responses (voxels, neurons, what-have-you).
        Each response should be Z-scored across time.
    Presp : array_like, shape (TP, M)
        Test responses with TP time points and M responses.
    alphas : list or array_like, shape (A,)
        Ridge parameters to be tested. Should probably be log-spaced. np.logspace(0, 3, 20) works well.
    normalpha : boolean
        Whether ridge parameters should be normalized by the Frobenius norm of Rstim. Good for
        comparing models with different numbers of parameters.
    dtype : np.dtype
        All data will be cast as this dtype for computation. np.single is used by default for memory
        efficiency.
    corrmin : float in [0..1]
        Purely for display purposes. After each alpha is tested, the number of responses with correlation
        greater than corrmin minus the number of responses with correlation less than negative corrmin
        will be printed. For long-running regressions this vague metric of non-centered skewness can
        give you a rough sense of how well the model is working before it's done.
    singcutoff : float
        The first step in ridge regression is computing the singular value decomposition (SVD) of the
        stimulus Rstim. If Rstim is not full rank, some singular values will be approximately equal
        to zero and the corresponding singular vectors will be noise. These singular values/vectors
        should be removed both for speed (the fewer multiplications the better!) and accuracy. Any
        singular values less than singcutoff will be removed.
    use_corr : boolean
        If True, this function will use correlation as its metric of model fit. If False, this function
        will instead use variance explained (R-squared) as its metric of model fit. For ridge regression
        this can make a big difference -- highly regularized solutions will have very small norms and
        will thus explain very little variance while still leading to high correlations, as correlation
        is scale-free while R**2 is not.
    Returns
    -------
    Rcorrs : array_like, shape (A, M)
        The correlation between each predicted response and each column of Presp for each alpha.
    
    """
    ## Calculate SVD of stimulus matrix
    logger.info("Doing SVD...")
    try:
        U,S,Vh = np.linalg.svd(Rstim, full_matrices=False)
    except np.linalg.LinAlgError as e:
        logger.info("NORMAL SVD FAILED, trying more robust dgesvd..")
        from text.regression.svd_dgesvd import svd_dgesvd
        U,S,Vh = svd_dgesvd(Rstim, full_matrices=False)

    ## Truncate tiny singular values for speed
    origsize = S.shape[0]
    ngoodS = np.sum(S>singcutoff)
    nbad = origsize-ngoodS
    U = U[:,:ngoodS]
    S = S[:ngoodS]
    Vh = Vh[:ngoodS]
    logger.info("Dropped %d tiny singular values.. (U is now %s)"%(nbad, str(U.shape)))

    ## Normalize alpha by the Frobenius norm
    #frob = np.sqrt((S**2).sum()) ## Frobenius!
    frob = S[0]
    #frob = S.sum()
    logger.info("Training stimulus has Frobenius norm: %0.03f"%frob)
    if normalpha:
        nalphas = alphas * frob
    else:
        nalphas = alphas

    ## Precompute some products for speed
    UR = np.dot(U.T, Rresp) ## Precompute this matrix product for speed
    PVh = np.dot(Pstim, Vh.T) ## Precompute this matrix product for speed
    
    #Prespnorms = np.apply_along_axis(np.linalg.norm, 0, Presp) ## Precompute test response norms
    zPresp = zs(Presp)
    Prespvar = Presp.var(0)
    Rcorrs = [] ## Holds training correlations for each alpha
    for na, a in zip(nalphas, alphas):
        #D = np.diag(S/(S**2+a**2)) ## Reweight singular vectors by the ridge parameter 
        D = S/(S**2+na**2) ## Reweight singular vectors by the (normalized?) ridge parameter
        
        pred = np.dot(mult_diag(D, PVh, left=False), UR) ## Best (1.75 seconds to prediction in test)
        # pred = np.dot(mult_diag(D, np.dot(Pstim, Vh.T), left=False), UR) ## Better (2.0 seconds to prediction in test)
        
        # pvhd = reduce(np.dot, [Pstim, Vh.T, D]) ## Pretty good (2.4 seconds to prediction in test)
        # pred = np.dot(pvhd, UR)
        
        # wt = reduce(np.dot, [Vh.T, D, UR]).astype(dtype) ## Bad (14.2 seconds to prediction in test)
        # wt = reduce(np.dot, [Vh.T, D, U.T, Rresp]).astype(dtype) ## Worst
        # pred = np.dot(Pstim, wt) ## Predict test responses

        if use_corr:
            #prednorms = np.apply_along_axis(np.linalg.norm, 0, pred) ## Compute predicted test response norms
            #Rcorr = np.array([np.corrcoef(Presp[:,ii], pred[:,ii].ravel())[0,1] for ii in range(Presp.shape[1])]) ## Slowly compute correlations
            #Rcorr = np.array(np.sum(np.multiply(Presp, pred), 0)).squeeze()/(prednorms*Prespnorms) ## Efficiently compute correlations
            Rcorr = (zPresp*zs(pred)).mean(0)
        else:
            ## Compute variance explained
            resvar = (Presp-pred).var(0)
            Rcorr = np.clip(1-(resvar/Prespvar), 0, 1)
            
        Rcorr[np.isnan(Rcorr)] = 0
        Rcorrs.append(Rcorr)
        
        log_template = "Training: alpha=%0.3f, mean corr=%0.5f, max corr=%0.5f, over-under(%0.2f)=%d"
        log_msg = log_template % (a,
                                  np.mean(Rcorr),
                                  np.max(Rcorr),
                                  corrmin,
                                  (Rcorr>corrmin).sum()-(-Rcorr>corrmin).sum())
        if logger is not None:
            logger.info(log_msg)
        else:
            print (log_msg)
    
    return Rcorrs

def bootstrap_ridge(Rstim, Rresp, alphas, nboots, chunklen, nchunks, dtype=np.single,
                    corrmin=0.2, joined=None, singcutoff=1e-10, normalpha=False, single_alpha=False,
                    use_corr=True, logger=logging.getLogger("ridge_corr")):
    """Uses ridge regression with a bootstrapped held-out set to get optimal alpha values for each response.
    [nchunks] random chunks of length [chunklen] will be taken from [Rstim] and [Rresp] for each regression
    run.  [nboots] total regression runs will be performed.  The best alpha value for each response will be
    averaged across the bootstraps to estimate the best alpha for that response.
    
    If [joined] is given, it should be a list of lists where the STRFs for all the voxels in each sublist 
    will be given the same regularization parameter (the one that is the best on average).
    
    Parameters
    ----------
    Rstim : array_like, shape (TR, N)
        Training stimuli with TR time points and N features. Each feature should be Z-scored across time.
    Rresp : array_like, shape (TR, M)
        Training responses with TR time points and M different responses (voxels, neurons, what-have-you).
        Each response should be Z-scored across time.
    alphas : list or array_like, shape (A,)
        Ridge parameters that will be tested. Should probably be log-spaced. np.logspace(0, 3, 20) works well.
    nboots : int
        The number of bootstrap samples to run. 15 to 30 works well.
    chunklen : int
        On each sample, the training data is broken into chunks of this length. This should be a few times 
        longer than your delay/STRF. e.g. for a STRF with 3 delays, I use chunks of length 10.
    nchunks : int
        The number of training chunks held out to test ridge parameters for each bootstrap sample. The product
        of nchunks and chunklen is the total number of training samples held out for each sample, and this 
        product should be about 20 percent of the total length of the training data.
    dtype : np.dtype
        All data will be cast as this dtype for computation. np.single is used by default for memory efficiency,
        as using np.double will thrash most machines on a big problem. If you want to do regression on 
        complex variables, this should be changed to np.complex128.
    corrmin : float in [0..1]
        Purely for display purposes. After each alpha is tested for each bootstrap sample, the number of 
        responses with correlation greater than this value will be printed. For long-running regressions this
        can give a rough sense of how well the model works before it's done.
    joined : None or list of array_like indices
        If you want the STRFs for two (or more) responses to be directly comparable, you need to ensure that
        the regularization parameter that they use is the same. To do that, supply a list of the response sets
        that should use the same ridge parameter here. For example, if you have four responses, joined could
        be [np.array([0,1]), np.array([2,3])], in which case responses 0 and 1 will use the same ridge parameter
        (which will be parameter that is best on average for those two), and likewise for responses 2 and 3.
    singcutoff : float
        The first step in ridge regression is computing the singular value decomposition (SVD) of the
        stimulus Rstim. If Rstim is not full rank, some singular values will be approximately equal
        to zero and the corresponding singular vectors will be noise. These singular values/vectors
        should be removed both for speed (the fewer multiplications the better!) and accuracy. Any
        singular values less than singcutoff will be removed.
    normalpha : boolean
        Whether ridge parameters (alphas) should be normalized by the Frobenius norm of Rstim. Good for rigorously
        comparing models with different numbers of parameters.
    single_alpha : boolean
        Whether to use a single alpha for all responses. Good for identification/decoding.
    use_corr : boolean
        If True, this function will use correlation as its metric of model fit. If False, this function
        will instead use variance explained (R-squared) as its metric of model fit. For ridge regression
        this can make a big difference -- highly regularized solutions will have very small norms and
        will thus explain very little variance while still leading to high correlations, as correlation
        is scale-free while R**2 is not.
    
    Returns
    -------
    wt : array_like, shape (N, M)
        Regression weights for N features and M responses.
    corrs : array_like, shape (M,)
        Validation set correlations. Predicted responses for the validation set are obtained using the regression
        weights: pred = np.dot(Pstim, wt), and then the correlation between each predicted response and each 
        column in Presp is found.
    alphas : array_like, shape (M,)
        The regularization coefficient (alpha) selected for each voxel using bootstrap cross-validation.
    bootstrap_corrs : array_like, shape (A, M, B)
        Correlation between predicted and actual responses on randomly held out portions of the training set,
        for each of A alphas, M voxels, and B bootstrap samples.
    valinds : array_like, shape (TH, B)
        The indices of the training data that were used as "validation" for each bootstrap sample.
    """
    nresp, nvox = Rresp.shape
    bestalphas = np.zeros((nboots, nvox))  ## Will hold the best alphas for each voxel
    valinds = [] ## Will hold the indices into the validation data for each bootstrap
    
    Rcmats = []
    for bi in counter(range(nboots), countevery=1, total=nboots):
        logger.info("Selecting held-out test set..")
        allinds = range(nresp)
        indchunks = list(zip(*[iter(allinds)]*chunklen))
        random.shuffle(indchunks)
        heldinds = list(itools.chain(*indchunks[:nchunks]))
        notheldinds = list(set(allinds)-set(heldinds))
        valinds.append(heldinds)
        
        RRstim = Rstim[notheldinds,:]
        PRstim = Rstim[heldinds,:]
        RRresp = Rresp[notheldinds,:]
        PRresp = Rresp[heldinds,:]
        
        ## Run ridge regression using this test set
        Rcmat = ridge_corr(RRstim, PRstim, RRresp, PRresp, alphas,
                           dtype=dtype, corrmin=corrmin, singcutoff=singcutoff,
                           normalpha=normalpha, use_corr=use_corr)
        
        Rcmats.append(Rcmat)
    
    ## Find weights for each voxel
    try:
        U,S,Vh = np.linalg.svd(Rstim, full_matrices=False)
    except np.linalg.LinAlgError as e:
        logger.info("NORMAL SVD FAILED, trying more robust dgesvd..")
        from text.regression.svd_dgesvd import svd_dgesvd
        U,S,Vh = svd_dgesvd(Rstim, full_matrices=False)

    ## Normalize alpha by the Frobenius norm
    #frob = np.sqrt((S**2).sum()) ## Frobenius!
    frob = S[0]
    #frob = S.sum()
    logger.info("Total training stimulus has Frobenius norm: %0.03f"%frob)
    if normalpha:
        nalphas = alphas * frob
    else:
        nalphas = alphas

    allRcorrs = np.dstack(Rcmats)
    if not single_alpha:
        logger.info("Finding best alpha for each response..")
        if joined is None:
            ## Find best alpha for each voxel
            meanbootcorrs = allRcorrs.mean(2)
            bestalphainds = np.argmax(meanbootcorrs, 0)
            valphas = nalphas[bestalphainds]
        else:
            ## Find best alpha for each group of voxels
            valphas = np.zeros((nvox,))
            for jl in joined:
                jcorrs = allRcorrs[:,jl,:].mean(1).mean(1) ## Mean across voxels in the set, then mean across bootstraps
                bestalpha = np.argmax(jcorrs)
                valphas[jl] = nalphas[bestalpha]
    else:
        logger.info("Finding single best alpha..")
        meanbootcorr = allRcorrs.mean(2).mean(1)
        bestalphaind = np.argmax(meanbootcorr)
        bestalpha = alphas[bestalphaind]
        valphas = np.array([bestalpha]*nvox)
        logger.info("Best alpha = %0.3f"%bestalpha)

    logger.info("Computing weights for each response using entire training set..")
    UR = np.dot(U.T, np.nan_to_num(Rresp))
    wt = np.zeros((Rstim.shape[1], Rresp.shape[1]))
    for ai,alpha in enumerate(nalphas):
        selvox = np.nonzero(valphas==alpha)[0]
        awt = reduce(np.dot, [Vh.T, np.diag(S/(S**2+alpha**2)), UR[:,selvox]])
        wt[:,selvox] = awt

    return wt, valphas, allRcorrs