# ==========
# GET SET UP
# ==========

import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from scipy.special import loggamma, digamma, logsumexp
from tqdm.notebook import trange, tqdm

# %%

# =============
# GENERATE DATA
# =============

def generateChains(nStates, nClusters):
    """
    generateChains randomly generates Markov chains for testing and also  initial probability distributions, q, both uniform on appropriate simplex

    inputs: nStates = #, alphabet size, labeled 0 thru nStates-1
            nClusters = #, number of distinct chains to generate

    outputs: transition_matrices = nClusters x nStates x nStates
             initDists = nClusters x nStates = q, 
    """    
    
    transition_matrices = np.random.exponential(scale=1.0,size=(nClusters, nStates, nStates))
    transition_matrices /= np.sum(transition_matrices, axis=-1)[..., np.newaxis]
    initDists = np.random.exponential(scale=1.0,size=(nClusters, nStates))
    initDists /= np.sum(initDists, axis=-1)[..., np.newaxis]

    return transition_matrices, initDists

def generateTrajectories(N, T, mixtureProbs, initDists, transition_matrices, fixed=False):
    """
    generateTrajectories is for testing. Generates samples a given mixture of Markov chains

    inputs: N = # of trajectories
            T = # of time points on each trajectory
            mixtureProbs = N x 1,  probabilities of each trajectory falling into a cluster
            initDists = nClusters x nStates, q, initial probability densities
            transition_matrices = nClusters x nStates x nStates 

    outputs: X = N x T, trajectories from alphabet nStates 
             trueLabels = N x 1,  true cluster labels for each trajectory
    """
    
    nClusters, nStates = np.shape(initDists)
    
    # initialize
    if fixed==False:
        trueLabels = np.random.choice(nClusters, N, p = mixtureProbs)
    else:
        trueLabels = np.zeros(N,)
        start_index = 0
        for i, fraction in enumerate(mixtureProbs):
            end_index = np.min([start_index + int(fraction * N),N])
            trueLabels[start_index:end_index] = i
            start_index = end_index  
        trueLabels =  trueLabels.astype('int64')
    cdf = np.cumsum(initDists, axis=-1)[trueLabels, :]
    U = np.random.uniform(size=N)[:, np.newaxis]
    now = np.argmax(U <= cdf, axis=-1)
    X = np.zeros(shape=(T, N), dtype=int)
    X[0, :] = now
    
    # loop over each time step
    cdf = np.cumsum(transition_matrices, axis=-1)
    for t in range(1, T):
        U = np.random.uniform(size=N)[:, np.newaxis]
        now = np.argmax(U <= cdf[trueLabels, now, ...], axis=-1)
        X[t, :] = now

    return X.T, trueLabels

# %%

# ======
# RUN EM
# ======    

# classical EM, previously implemented to benchmark against. Not used in text. 
def doEM(X, M, nStates, tol=1e-12, max_iters=1000):

    # X0 = N x nStates matrix, indicates initial conditions of N trajectories
    X0 = (X[:, 0, np.newaxis] == np.arange(nStates)[np.newaxis, :])
    
    # transitions = N x nStates x nStates matrix, counts transitions of N trajectories
    first = X[:, :-1, np.newaxis] == np.arange(nStates)[np.newaxis, np.newaxis, :]
    second = X[:, 1:, np.newaxis] == np.arange(nStates)[np.newaxis, np.newaxis, :]

    transitions = first[..., np.newaxis] * second[..., np.newaxis, :]
    transitions = np.sum(transitions, axis = 1)

    # first E step is a random soft assignment
    zHat = np.random.exponential(scale=1.0,size=(np.shape(X)[0], M)) # for
    zHat /= np.sum(zHat, axis=1)[:, np.newaxis]
    
    for steps in range(max_iters):

        # perform M step
        muHat = np.mean(zHat, axis=0)
        probs = zHat.T / zHat.sum(axis=0)[:, np.newaxis]
        qHat = probs @ X0
        pHat = np.tensordot(probs, transitions, axes = (1, 0))
        np.divide(pHat, (np.sum(pHat, axis = -1)[..., np.newaxis]), 
                  pHat, where=(pHat > 0))

        # calculate log-likelihoods (with proper treatment of the 0 case)
        logprobs = muHat[np.newaxis, :] * (X0 @ qHat.T)
        zeros = (logprobs <= 0)
        np.log(logprobs, logprobs, where = ~zeros)
        updates = pHat[np.newaxis, ...]
        zero_updates = (updates <= 0)
        np.log(updates, updates, where = ~zero_updates)
        logprobs += np.sum(transitions[:, np.newaxis, ...] * updates, axis=(-2,-1))
        zeros = zeros + np.sum(transitions[:, np.newaxis, ...] * zero_updates, axis=(-2,-1))
        zeros = (zeros  > 0)
        logprobs[zeros] = -np.inf
        
        # perform E step

        zHat = logprobs -logsumexp(logprobs, axis=1)[:, np.newaxis]

        logL = np.sum(logsumexp(logprobs + zHat, axis = 1))
        zHat = np.exp(zHat)

        # check tolerance and possibly break
        #print('Step: ', steps, '\nLog likelihood: ', logL)
        if steps == 0:
            muHat_old = muHat
            qHat_old = qHat
            pHat_old = pHat
        else:
            diff = np.sum((muHat - muHat_old)**2)
            diff += np.sum((qHat - qHat_old)**2)
            diff += np.sum((pHat - pHat_old)**2)
            if diff ** .5 < tol:
                break
            muHat_old = muHat
            qHat_old = qHat
            pHat_old = pHat

    return zHat, muHat, qHat, pHat, steps, logL

# %%

# ======
# RUN VEM
# ======    

def doVEM(X, M, nStates, tol=1e-12, max_iters=1000,alpha=None):
    """
    doVEM performs a single run of our VEM algorithm on trajectories X, stopping when the next step hits a level of tolerance or a max number of iterations.

    inputs: X = N x T matrix, N trajectories of length T, 
            M = #, number of clusters (k in the text)
            tol  =  absolute err level; when log-likelihood stops changing by this amount * N * T, halt the algorithm, default 1e-12
            max_iters = maximum number of iterations, default 1000
            alpha = cluster # hyperparameter. Default is 1/k. See text for discussion. 


    outputs: zHat = updated estimates of probability of each cluster
             muHat = nClusters x 1, estimated mu, mixture probs 
             qHat = nClusters x nStates, estimated initial probs
             pHat = nClusters x nStates x nStates, estimated transition matrices
             steps = # of steps EM alg took
             logL = ELBO at termination
    """
    
    if alpha==None:
        alpha=1.0/M
     
    N = np.shape(X)[0]
    T = np.shape(X)[1]
    
    # X0 = N x nStates matrix, indicates initial conditions of N trajectories
    X0 = (X[:, 0, np.newaxis] == np.arange(nStates)[np.newaxis, :])
    
    # transitions = N x nStates x nStates matrix, counts transitions of N trajectories
    first = X[:, :-1, np.newaxis] == np.arange(nStates)[np.newaxis, np.newaxis, :]
    second = X[:, 1:, np.newaxis] == np.arange(nStates)[np.newaxis, np.newaxis, :]

    first = first[..., np.newaxis]
    second = second[..., np.newaxis, :]
    
    #transitions = np.multiply(first,second)
    #transitions = np.sum(transitions, axis = 1)

    # added by Chris -- more memory efficient but slower.
    # uncomment/replace previous matrix multiply for speed
    transitions_brute = np.zeros([N,nStates,nStates])
    for n in range(N):
        transitions_brute[n,:,:] = np.sum(np.multiply(first[n,:,:,:],second[n,:,:,:]),axis=0)
    transitions = transitions_brute


    # first E step is a random soft assignment
    zHat = np.random.exponential(scale=1.0,size=(np.shape(X)[0], M))
    zHat /= np.sum(zHat, axis=1)[:, np.newaxis]
    
    for steps in range(max_iters):

        # perform M step, components
        muChange = np.sum(zHat, axis=0)
        muHat = muChange + alpha
        logMuTilde = digamma(muHat) - digamma(np.sum(muHat))
        priorChange = muChange @ logMuTilde + M * loggamma(alpha) - loggamma(M * alpha)
        priorChange -= (np.sum(loggamma(muHat)) - loggamma(np.sum(muHat)))
        if steps == 0:
            muSave = np.copy(muHat)[np.newaxis, :]
        else:
            muSave = np.row_stack((muSave, muHat))
        
        # perform M step, initial states
        qChange = zHat.T @ X0
        qHat = qChange + 1.0

        logQTilde = digamma(qHat) - digamma(np.sum(qHat, axis=1))[:, np.newaxis]
        priorChange += np.sum(qChange * logQTilde) + M * nStates * loggamma(1)
        priorChange -= (np.sum(loggamma(qHat)) - np.sum(loggamma(np.sum(qHat, axis=1))))

        # perform M step, transitions
        pChange = np.tensordot(zHat.T, transitions, axes = (1, 0))
        pHat = pChange + 1.0

        logPTilde = digamma(pHat) - digamma(np.sum(pHat, axis = 2))[:, :, np.newaxis]   
        priorChange += np.sum(pChange * logPTilde) + M * nStates ** 2 * loggamma(1)
        priorChange -= (np.sum(loggamma(pHat)) - np.sum(loggamma(np.sum(pHat, axis=2))))

        # perform E stap
        logprobs = logMuTilde[np.newaxis, :] + (X0 @ logQTilde.T)
        logprobs += np.tensordot(transitions, logPTilde, axes = ([1, 2], [1, 2]))
        logConstants = logsumexp(logprobs, axis=1)
        zHat = np.exp(logprobs - logConstants[:, np.newaxis])
        
        # check tolerance and possibly break
        if steps == 0:
            logL = -priorChange + np.sum(logConstants)
        else:
            oldL = logL
            logL = -priorChange + np.sum(logConstants)
            diff = np.abs(oldL-logL)
            if diff < tol*N*T:
                break
    return zHat, muHat, qHat, pHat, steps, logL

##

# multi-start doVEM. Outputs run corresponding to largest ELBO.
def doVEMmulti(X, M, nStates, tol=1e-12, max_iters=1000,alpha=None,nEM=100):

    if alpha==None:
        alpha=1.0/M

    logL_best = float('-inf')
    for s in trange(nEM):
        zHat, muHat, qHat, pHat, steps, logL= doVEM(X, M, nStates, tol, max_iters=max_iters,alpha=alpha)
        if logL>=logL_best:    
            zHat_best = zHat
            muHat_best = muHat
            qHat_best = qHat
            pHat_best = pHat
            logL_best = logL
            stepsBest = steps
    return zHat_best, muHat_best, qHat_best, pHat_best,stepsBest, logL_best        

##

# parallelized version of doVEmulti
def doVEMmultiPar(X, M, nStates, tol=1e-12, max_iters=1000,alpha=None, nEM=100):
    
    if alpha==None:
        alpha=1.0/M

    from joblib import Parallel, delayed
 
    parallel_gen =  Parallel(n_jobs=-1, return_as="generator_unordered",verbose=1)(delayed(doVEM)(X, M, nStates, tol, max_iters=max_iters,alpha=alpha) for _ in range(nEM))    


    logL_best = float('-inf')
    for s in parallel_gen:
        zHat, muHat, qHat, pHat, steps, logL= s
        if logL>=logL_best:    
            zHat_best = zHat
            muHat_best = muHat
            qHat_best = qHat
            pHat_best = pHat
            logL_best = logL
            stepsBest = steps
    return zHat_best, muHat_best, qHat_best, pHat_best,stepsBest, logL_best        


##
def doEMmulti(X, M, nStates, tol=1e-12, nEM=100, max_iters=1000):

    logL_best = float('-inf')

    for s in range(nEM):
        zHat, muHat, qHat, pHat, steps, logL= doEM(X, M, nStates, tol, max_iters=max_iters)
        if logL>=logL_best:    
            zHat_best = zHat
            muHat_best = muHat
            qHat_best = qHat
            pHat_best = pHat
            logL_best = logL
            stepsBest = steps
    return zHat_best, muHat_best, qHat_best, pHat_best,stepsBest, logL_best        


# %%%%

# =================
# CHECK ERROR AFTER
# =================

def find_best_clustermatch(trueLabels, zHat, muHat, qHat, pHat):
    """
    find_best_clustermatch uses the Hungarian algorithm to find the permutation of zHat that minimizes the error with the true Z

    inputs: trueLabels: N x 1 array of true cluster labels, e.g. [0,1,2,...]
            zHat: N x nCluster matrix of estimated Z probabilities
            muHat, qHat, pHat: all estimates from EM, no effect on algorithm

    outputs: trueLabels = nCluster x 1 array of optimal permutation,
             zHat_c, qHat_c, muHat_c, pHat_c = original versions permuted to optimal 
    """     
    
    # set up
    N, nClusters = np.shape(zHat)    
    zTrue = np.zeros((N, nClusters))
    zTrue[np.arange(N), trueLabels] = 1.

    # run Hungarian algorithm
    cost = zTrue.T @ zHat
    _, best_assignment = linear_sum_assignment(cost, maximize=True)

    # reorder estimates
    zHat_c = zHat[..., best_assignment]
    muHat_c = muHat[best_assignment]
    qHat_c = qHat[best_assignment, ...]
    pHat_c = pHat[best_assignment, ...]
    
    return best_assignment, zHat_c, qHat_c, muHat_c, pHat_c



