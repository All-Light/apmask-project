"""
===============================================================================
Bayesian Inference in Diffusion Tensor Imaging - Project Template
===============================================================================

This Python file provides the starter template for the course project in
"Advanced Probabilistic Machine Learning",
Department of Information Technology, Uppsala University.

Authors:
- Jens Sjölund (original author) - jens.sjolund@it.uu.se
- Anton O'Nils (updates & finalization) - anton.o-nils@it.uu.se
- Stina Brunzell (updates & finalization) - stina.brunzell@it.uu.se

------------------------------------------------------------------------------
Purpose
------------------------------------------------------------------------------
The project concerns Bayesian inference in diffusion MRI (dMRI), specifically
the diffusion tensor model (DTI). The goal is to estimate local tissue
properties (baseline signal S0 and diffusion tensor D) from real-world dMRI
measurements, using different Bayesian inference techniques.

Each student/group member will implement one of the following inference methods:
  1. Metropolis-Hastings  
  2. Importance Sampling  
  3. Variational Inference  
  4. Laplace Approximation  

The provided code gives:
  - Utilities for loading and preprocessing the "Stanford HARDI dataset". 
  - Helper functions for matrix operations, parameterizations and gradients.  
  - A skeleton structure for the prior, likelihood, and posterior approx.   
  - Placeholders where each inference method should be implemented.  
  - Plotting routines to visualize posterior summaries.

------------------------------------------------------------------------------
Dataset
------------------------------------------------------------------------------
The code uses the Stanford HARDI diffusion MRI dataset (Rokem et al., 2015),
accessible via DIPY's "get_fnames('stanford_hardi')".

------------------------------------------------------------------------------
Notes
------------------------------------------------------------------------------
- Several classes and methods are left as "NotImplementedError"; students are
  expected to fill these in.  
- Computations are memoized with "disk_memoize" to avoid repeated costly runs.  
- Results for each inference method are automatically plotted and saved.  

=============================================================================
Imports
=============================================================================
Required libraries: numpy, matplotlib, scipy, dipy
Install with: pip install numpy matplotlib scipy dipy
"""

# Standard library: general utilities
import os
import pickle
import hashlib
from functools import wraps

# NumPy and Matplotlib: math and plotting
import numpy as np
import matplotlib.pyplot as plt

# SciPy: probability distributions, math functions, and optimization
# Hint: these tools might be useful later in the project
from scipy.stats import gamma, norm, wishart, multivariate_normal
from scipy.spatial.transform import Rotation
from scipy.special import logsumexp, digamma
from scipy.optimize import minimize

# DIPY: diffusion MRI utilities and models
from dipy.io.image import load_nifti, save_nifti   # for loading / saving imaging datasets
from dipy.io.gradients import read_bvals_bvecs     # for loading / saving our bvals and bvecs
from dipy.core.gradients import gradient_table     # for constructing gradient table from bvals/bvecs
from dipy.data import get_fnames                   # for small datasets that we use in tests and examples
from dipy.segment.mask import median_otsu          # for masking out the background
import dipy.reconst.dti as dti                     # for diffusion tensor model fitting and metrics

# weighted credible interval
from statsmodels.stats.weightstats import DescrStatsW


"""
=============================================================================
Caching Utility (already implemented)
=============================================================================
Provides disk-based memoization to avoid recomputation.
"""

def disk_memoize(cache_dir="cache"):
    """
    Decorator for caching function outputs on disk.

    This utility is already implemented and should not be modified by students.
    It allows expensive computations to be stored and re-used across runs,
    based on the function arguments. If you call the same function again with
    the same inputs, it returns the cached results instead of recomputing.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Optionally force a fresh computation (ignores cache if True)
            force = kwargs.pop("force_recompute", False)

            # Make sure the cache directory exists
            os.makedirs(cache_dir, exist_ok=True)

            # Build a unique hash key from the function name and arguments
            func_name = func.__name__
            key = (func_name, args, kwargs)
            hash_str = hashlib.md5(pickle.dumps(key)).hexdigest()
            cache_path = os.path.join(cache_dir, f"{func_name}_{hash_str}.pkl")

            # Load the cached result if it exists (and recomputation is not forced)
            if not force and os.path.exists(cache_path):
                with open(cache_path, "rb") as f:
                    return pickle.load(f)

            # Otherwise: compute the result, then cache it to disk
            result = func(*args, **kwargs)
            with open(cache_path, "wb") as f:
                pickle.dump(result, f)

            return result
        
        return wrapper
    return decorator



"""
=============================================================================
Data Loading & Preprocessing (already implemented)
=============================================================================
Loads the Stanford HARDI dataset, applies masking/cropping, and
extracts one voxel with a DTI point estimate for testing.
"""

@disk_memoize()
def get_preprocessed_data():
    """
    Load and preprocess a single voxel of diffusion MRI data.

    What it does:
    - Loads the dataset and gradient information (b-values and b-vectors).
    - Fits a diffusion tensor model (DTI) to one voxel.
    - Extracts a point estimate: baseline signal (S0), eigenvalues, eigenvectors.

    Returns
    -------
    y : ndarray
        Observed diffusion MRI signal vector for a single voxel.
    point_estimate : [S0, evals, evecs]
        Estimated baseline signal, eigenvalues, and eigenvectors.
    gtab : GradientTable
        Gradient table with b-values (diffusion weighting strength)
        and b-vectors (gradient directions).
    """

    # Load the masked data, background mask, and gradient information
    data, mask, gtab = get_data()

    # Initialize a diffusion tensor model (DTI) with S0 estimation enabled
    tenmodel = dti.TensorModel(gtab, return_S0_hat=True)

    # Extract the signal for a single voxel (coordinates chosen for this project)
    y = data[35, 35, 30, :]

    # Fit the DTI model to this voxel's signal
    tenfit = tenmodel.fit(y)
    
    # Extract point estimates: baseline signal, eigenvalues, and eigenvectors
    S0 = tenfit.S0_hat
    evals = tenfit.evals
    evecs = tenfit.evecs
    point_estimate = [S0, evals, evecs]

    # Return the raw voxel signal, point estimate, and gradient table
    return y, point_estimate, gtab


def get_data():
    """
    Load and preprocess the Stanford HARDI diffusion MRI dataset.

    What it does:
    - Downloads the dataset if not already present (via DIPY).
    - Loads the 4D diffusion MRI volume (x, y, z, measurements).
    - Reads b-values (diffusion weighting strength) and b-vectors (gradient directions).
    - Creates a gradient table (gtab) combining this information.
    - Applies a brain mask and cropping to remove background and reduce size.

    Returns
    -------
    maskdata : ndarray
        The masked and cropped diffusion MRI data.
    mask : ndarray (boolean)
        The brain mask used to exclude background voxels.
    gtab : GradientTable
        Gradient information (b-values and b-vectors) for each measurement.
    """

    # Download filenames for the Stanford HARDI dataset if not already cached 
    hardi_fname, hardi_bval_fname, hardi_bvec_fname = get_fnames('stanford_hardi')

    # Load the raw 4D dataset: dimensions are (x, y, z, diffusion measurements)
    data, _ = load_nifti(hardi_fname)

    # Read diffusion weighting information (b-values, b-vectors) and build gradient table
    bvals, bvecs = read_bvals_bvecs(hardi_bval_fname, hardi_bvec_fname)
    gtab = gradient_table(bvals, bvecs)

    # Apply brain masking and cropping to remove background and save compute
    maskdata, mask = median_otsu(
        data, vol_idx=range(10, 50), median_radius=3, numpass=1, autocrop=True, dilate=2
    )

    # Print the final data shape for confirmation
    print('Loaded data with shape: (%d, %d, %d, %d)' % maskdata.shape)

    return maskdata, mask, gtab


"""
=============================================================================
Linear Algebra Helpers (already implemented)
=============================================================================
Functions for reconstructing tensors and switching between
parameterizations. Already implemented.
Hint: you will make use of these helpers later in the project,
the ones involving theta are useful for VI and Laplace.
"""

def compute_D(evals, V):
    """
    Reconstruct the diffusion tensor D from eigenvalues and eigenvectors.

    D = V Λ V.T, where Λ is the diagonal matrix of eigenvalues.

    Parameters
    ----------
    evals : ndarray
        Eigenvalues, shape (3,) or batched.
    V : ndarray
        Eigenvectors, shape (3, 3) or batched.

    Returns
    -------
    D : ndarray
        Diffusion tensor(s), shape (..., 3, 3).
    """

    # Ensure inputs have the correct batch dimensions
    if evals.ndim == 1:
        evals = evals[None, None, :]
    elif evals.ndim == 2:
        evals = evals[:, None, :]
    if V.ndim == 2:
        V = V[None, :, :]

    # Compute D = V Λ V.T as V (V @ Λ).T
    V_scaled = V * evals
    D = np.matmul(V, np.transpose(V_scaled, axes=[0, 2, 1]))

    return D


def theta_from_D(D):
    """
    Convert a diffusion tensor D into an unconstrained parameter vector theta.

    Follows Eq. (18): D = L L.T with L from Cholesky factorization.
    Diagonals are log-transformed, off-diagonals kept raw.

    Parameters
    ----------
    D : ndarray (3, 3)
        Symmetric positive-definite diffusion tensor.

    Returns
    -------
    theta : ndarray (6,)
        Unconstrained parameter vector corresponding to the lower-triangular
        entries of L (log of diagonals, raw off-diagonals).
    """
    
    # Compute Cholesky factor (lower-triangular L) of D
    L = np.linalg.cholesky(D)
    
    # Indices of lower-triangular entries (including diagonal)
    p = D.shape[0]
    tril_indices = np.tril_indices(p)
    theta = []

    # Store log of diagonal entries, raw off-diagonal entries
    for i, j in zip(*tril_indices):
        if i == j:
            theta.append(np.log(L[i, j]))   # Diagonal: log-transform
        else:
            theta.append(L[i, j])           # Off-diagonal: raw value

    return np.array(theta)


def D_from_theta(theta):
    """
    Convert unconstrained parameter vector theta back into diffusion tensor D.

    Follows Eq. (18): D = L L.T with L constructed from theta.
    Diagonal entries of L are exponentiated to ensure positivity,
    off-diagonals are used as raw values.

    Parameters
    ----------
    theta : ndarray (..., 6)
        Unconstrained parameters corresponding to the lower-triangular
        entries of L (log-diagonals, raw off-diagonals).

    Returns
    -------
    D : ndarray (..., 3, 3)
        Symmetric positive-definite diffusion tensor(s).
    """
    
    # Ensure theta is an array and check shape
    theta = np.asarray(theta)
    *batch_shape, _ = theta.shape
    assert theta.shape[-1] == 6, "Last dimension must be 6 for 3x3 lower-triangular matrices."

    # Initialize lower-triangular matrix L
    L = np.zeros((*batch_shape, 3, 3), dtype=theta.dtype)

    # Fill L with exponentiated diagonals and raw off-diagonals
    tril_indices = np.tril_indices(3)
    for k, (i, j) in enumerate(zip(*tril_indices)):
        if i == j:
            L[..., i, j] = np.exp(theta[..., k])   # Diagonal
        else:
            L[..., i, j] = theta[..., k]           # Off-diagonal

    # Reconstruct D = L @ L.T (batch-aware matrix multiplication)
    D = L @ np.swapaxes(L, -1, -2)

    return D.squeeze()


def grad_D_wrt_theta_at_D(D):
    """
    Compute nabla_theta D evaluated at D.

    Uses the parameterization in Eq. (18): D = L L.T where L is built from 
    theta. Returns the gradient tensor with one (3x3) slice per theta component.

    Parameters
    ----------
    D : ndarray (3, 3)
        Symmetric positive-definite diffusion tensor.

    Returns
    -------
    grad_D : ndarray (3, 3, 6)
        Gradient of D w.r.t. theta, one 3x3 matrix per parameter.
    """
    
    # Get Cholesky factor of D and set up indices for lower-triangular entries
    p = D.shape[0]
    L = np.linalg.cholesky(D)
    tril_indices = np.tril_indices(p)
    num_params = len(tril_indices[0])

    # Prepare output container
    grad_D = np.zeros((p, p, num_params))

    # Loop over all parameters in theta
    for k, (m, n) in enumerate(zip(*tril_indices)):

        # Build a basis matrix for the effect of this parameter
        E_mn = np.zeros((p, p))
        if m == n:
            # Diagonal: dL_mm/dtheta = L_mm since L_mm = exp(theta)
            factor = L[m, n]
        else:
            # Off-diagonal: dL_mn/dtheta = 1
            factor = 1.0
        E_mn[m, n] = factor

        # Work out the corresponding change in D
        dD_k = E_mn @ L.T + L @ E_mn.T
        grad_D[:, :, k] = dD_k

    return grad_D


"""
=============================================================================
Bayesian Model Components (need to be implemented)
=============================================================================
Students: implement all parts in this section (priors, likelihoods, etc.)
These are required before any inference method can be attempted.
"""

# Goal p(Diffussion, S0 | Data)

class frozen_prior:
    def __init__(self, alpha_s, theta_s, alpha_lambda, theta_lambda):
        self.p_s0 = gamma(a=alpha_s, scale=theta_s)
        self.p_lam1 = gamma(a=alpha_lambda, scale=theta_lambda)
        self.p_lam2 = gamma(a=alpha_lambda, scale=theta_lambda)
        self.p_lam3 = gamma(a=alpha_lambda, scale=theta_lambda)

    # random variable sampling
    def rvs(self, size):
        #p(z) = p(S0) p(λ1) p(λ2) p(λ3) p(V)
        s0 = self.p_s0.rvs(size=size)
        lam1 = self.p_lam1.rvs(size=size)
        lam2 = self.p_lam2.rvs(size=size)
        lam3 = self.p_lam3.rvs(size=size)
        V = Rotation.random(size).as_matrix()
        return s0, lam1,lam2,lam3, V
    
    def logpdf(self, s0, lam1, lam2, lam3, V):
        return self.p_s0.logpdf(s0) + self.p_lam1.logpdf(lam1) + self.p_lam2.logpdf(lam2) + self.p_lam3.logpdf(lam3)


class frozen_likelihood:
    def __init__(self, gtab, sigma):        
        self.gtab = gtab   # store gradient table with b-values and b-vectors
        self.sigma = sigma

    def logpdf(self, S0, evecs, evals, y):
        #p(z|Data) \prop p(Data|z)p(z)
        S0 = np.atleast_1d(S0)        # ensure S0 is array-like
        D = compute_D(evals, evecs)   # reconstruct diffusion tensor

        # Build q from diffusion gradients (b-values & b-vectors),
        # corresponds to the experimental setting x in the project instructions
        q = np.sqrt(self.gtab.bvals[:, None]) * self.gtab.bvecs

        # Model signal S given tensor D and baseline S0
        S = S0[:, None] * np.exp( - np.einsum('...j, ijk, ...k->i...', q, D, q))

        return np.sum(norm.logpdf(y, loc=S, scale=self.sigma))
        


"""
=============================================================================
Posterior Approximations (need to be implemented)
=============================================================================
Students: implement these approximations, which are only used in the
corresponding inference methods below:
  - variational_posterior: used only for Variational Inference
  - mvn_reparameterized: used only for Laplace Approximation

They are NOT needed for Metropolis-Hastings or Importance Sampling.
"""

class variational_posterior:
    # Placeholder for variational posterior approximation.
    # Hint: you may want to add input parameters to these methods.
    # The score() method is already implemented and can be used later
    # when implementing inference (with REINFORCE leave-one-out estimator).

    def __init__(self):
        raise NotImplementedError

    def logpdf(self):
        raise NotImplementedError
    
    def rvs(self, size):
        raise NotImplementedError

        return S0_samples, evals_samples, evecs_samples

    def score(self, S0, D):
        # Combine score contributions from gamma and Wishart parts
        score_wrt_log_shape, score_wrt_log_scale = self.gamma_score(S0)
        score_wrt_theta, score_wrt_log_df = self.wishart_score(D)
        return np.concatenate([
            score_wrt_log_shape, score_wrt_log_scale, score_wrt_theta, score_wrt_log_df]
        )

    def gamma_score(self, x):
        # Score function for gamma distribution
        score_wrt_log_shape = (np.log(x / self.scale) - digamma(self.shape)) * self.shape
        score_wrt_log_scale = (x / self.scale**2 - self.shape / self.scale) * self.scale
        return score_wrt_log_shape, score_wrt_log_scale

    def wishart_score(self, D):
        # Score function for Wishart distribution
        W = self.df * D
        Sigma_inv = np.linalg.inv(self.Sigma)
        score_wrt_Sigma = 0.5 * Sigma_inv @ (W - self.df * self.Sigma) @ Sigma_inv
        score_wrt_theta = np.tensordot(
            score_wrt_Sigma, grad_D_wrt_theta_at_D(self.Sigma), axes=([0,1], [0,1])
        )
        p = W.shape[0]
        _, logdet_W = np.linalg.slogdet(W)
        _, logdet_Sigma = np.linalg.slogdet(self.Sigma)
        digamma_sum = np.sum([digamma((self.df + 1 - j) / 2.0) for j in range(1, p+1)])
        score_wrt_log_df = ((self.df - 2) / 2) * (logdet_W - p * np.log(2) - logdet_Sigma - digamma_sum)
        return score_wrt_theta, score_wrt_log_df


class mvn_reparameterized:
    """
    Multivariate Normal distribution in the transformed parameter space theta in R^7,
    where:
      - theta[0] = log(S0)
      - theta[1:7] = unconstrained lower-triangular Cholesky entries of D.
    """
    def __init__(self, mean, cov):
        self.mean = np.asarray(mean)
        self.cov = np.asarray(cov)
        self.mvn = multivariate_normal(mean=self.mean, cov=self.cov)

    def logpdf(self, theta):
        """Evaluate log-density of theta in the transformed parameter space."""
        return self.mvn.logpdf(theta)

    def rvs(self, size=1):
        """
        Sample theta ~ N(mean, cov) and transform back to (S0, evals, evecs).

        Returns
        -------
        S0_samples : ndarray, shape (size,)
        evals_samples : ndarray, shape (size, 3)
        evecs_samples : ndarray, shape (size, 3, 3)
        """
        theta_samples = self.mvn.rvs(size=size)
        if size == 1:
            theta_samples = theta_samples[None, :]

        # 1. Map log(S0) -> S0
        S0_samples = np.exp(theta_samples[:, 0])

        # 2. Map theta[1:] -> Diffusion Tensor D
        D_samples = D_from_theta(theta_samples[:, 1:])

        # 3. Eigendecomposition to get eigenvalues and eigenvectors
        # np.linalg.eigh handles batched (size, 3, 3) arrays efficiently
        evals_samples, evecs_samples = np.linalg.eigh(D_samples)

        return S0_samples, evals_samples, evecs_samples


"""
=============================================================================
Inference Methods (need to be implemented)
=============================================================================
Students: implement one method each (MH, IS, VI, or Laplace).
Uses memoization to speed up repeated runs.
"""

@disk_memoize()
def metropolis_hastings(y, n_samples, prior, likelihood, gamma_param, nu_param, S0_init, D_init, plot_traces=False):
    # Students: implement Metropolis-Hastings here.
    # Before starting, make sure the prior and likelihood are implemented.
    # Note: you may change, add, or remove input parameters depending on your design
    # (e.g. pass initialization values like those prepared in main()).

    # q_S0 = gamma(a=gamma_param ** (-2), scale=gamma_param ** 2 * S0_init)  # gamma distribution
    # q_D = wishart(df=nu_param, scale=D_init / nu_param)  # wishart distribution
    rng = np.random.default_rng()

    def log_target(S0, D):
        evals, evecs = np.linalg.eigh(D)

        log_likelihood = likelihood.logpdf(
            S0=S0, evals=evals, evecs=evecs, y=y,
        )
        log_prior = prior.logpdf(
            s0=S0,
            lam1=evals[0],
            lam2=evals[1],
            lam3=evals[2],
            V=evecs,
        )

        # Convert the eigenvalue/rotation prior density to a density over D
        log_jacobian = (
                np.log(evals[1] - evals[0])
                + np.log(evals[2] - evals[0])
                + np.log(evals[2] - evals[1])
        )

        return log_likelihood + log_prior - log_jacobian

    def log_proposal(S0_to, D_to, S0_from, D_from):
        return (
                gamma(
                    a=gamma_param ** (-2),
                    scale=gamma_param ** 2 * S0_from,
                ).logpdf(S0_to)
                + wishart(
            df=nu_param,
            scale=D_from,
        ).logpdf(D_to)
        )

    S0_current = S0_init
    D_current = D_init

    S0_samples, evals_samples, evecs_samples = np.zeros(n_samples), np.zeros((n_samples, 3)), np.zeros((n_samples, 3, 3))

    for i in range(n_samples):
        S0_proposed = gamma(
            a=gamma_param ** (-2),
            scale=gamma_param ** 2 * S0_current,
        ).rvs()

        D_proposed = wishart(df=nu_param, scale=D_current).rvs()

        log_a = (
                log_target(S0_proposed, D_proposed)
                + log_proposal(S0_current, D_current, S0_proposed, D_proposed)  # reverse

                - log_target(S0_current, D_current)
                - log_proposal(S0_proposed, D_proposed, S0_current, D_current)  # forward
        )

        if np.log(rng.uniform()) < min(0.0, log_a):
            S0_current = S0_proposed
            D_current = D_proposed

        S0_samples[i] = S0_current
        evals, evecs = np.linalg.eigh(D_current)
        evals_samples[i] = evals
        evecs_samples[i] = evecs


    return S0_samples, evals_samples, evecs_samples


@disk_memoize()
def importance_sampling(prior, likelihood, y, n_samples, gamma_param, nu_param, S0_init, D_init):
    # Students: implement Importance Sampling here.
    # Before starting, make sure the prior and likelihood are implemented.
    # Note: you may change, add, or remove input parameters depending on your design
    # (e.g. pass initialization values like those prepared in main()).

    # problem: We cannot sample the true pdf pi(x) of our problem and must estimate it
    # main idea:
    #   - sample a proposal r(x) (wishart)
    #   - compensate the mismatch by importance weights wi

    q_s0 = gamma(a=gamma_param**(-2), scale=gamma_param**2*S0_init) # gamma distribution
    q_D = wishart(df=nu_param, scale=D_init/nu_param) # wishart distribution

    S0_samples = q_s0.rvs(size=n_samples)
    D_samples = q_D.rvs(size=n_samples)

    evals_samples = np.zeros((n_samples,3)) #eigenvalues
    evecs_samples = np.zeros((n_samples,3,3)) #eigenvectors

    importance_weights = np.zeros(n_samples)
    for n in range(n_samples):

        evals, evecs = np.linalg.eigh(D_samples[n])
        evals_samples[n] = evals
        evecs_samples[n] = evecs

        # correction term due to the wishart being sampled 
        #  using other coordinates
        log_jacobian = (
            np.log(evals[1] - evals[0])
            + np.log(evals[2] - evals[0])
            + np.log(evals[2] - evals[1])
        )
        # likelihood sample
        log_likelihood = likelihood.logpdf(
            S0=S0_samples[n], 
            evals=evals,
            evecs=evecs, 
            y=y
        )
        # prior sample
        log_prior = prior.logpdf(
            s0=S0_samples[n],
            lam1=evals[0],
            lam2=evals[1],
            lam3=evals[2],
            V=evecs
        )

        log_proposal = (
            q_s0.logpdf(S0_samples[n])
            + q_D.logpdf(D_samples[n])
            + log_jacobian
        )
        # weight = likelihood*prior/proposal pdfs
        importance_weights[n] = (
            log_likelihood
            + log_prior
            - log_proposal
        )


    return importance_weights, S0_samples, evals_samples, evecs_samples  #S0_samples, evals_samples, evecs_samples

@disk_memoize()
def sequential_monte_carlo_sampling(prior, likelihood, y, n_samples, gamma_param, nu_param, S0_init, D_init):
    # Importance sampling jumps directly to from proposal to posterior, in SMCS we move in internediate distributions
    # z = (S0, lam1, lam2, lam 3, V)
    # p(z|y) \prop p(z)p(y|z)
    # pi_t(z) \prop p(z)p(y|z)^(beta)
    s0, lam1, lam2, lam3, V = prior.rvs(size=n_samples)

    log_weights = np.full(n_samples, -np.log(n_samples))
    evals = np.array([lam1,lam2,lam3]).T
    log_likelihoods = np.array([likelihood.logpdf(
        S0=s0[i], 
        evals=evals[i],
        evecs=V[i], 
        y=y
    ) for i in range(n_samples)], dtype=float)

    prev_beta = 0.0
    betas = np.linspace(0,1,1001)
    for beta in betas:
        log_weights += (beta-prev_beta) *log_likelihoods
        log_weights -= logsumexp(log_weights)

        weights = np.exp(log_weights)
        effective_sample_size = 1/np.sum(weights**2)
        #print(f"effective sample size (N_ESS) before resampling beta={beta}: {effective_sample_size} out of total {n_samples}")
        if(effective_sample_size < 0.8*n_samples):
            # resampling if we have too low ess
            indices = np.random.choice(
                n_samples, size=n_samples, replace=True, p=weights
            )
            s0 = s0[indices]
            evals = evals[indices]
            V = V[indices]
            log_likelihoods = log_likelihoods[indices]
            log_weights.fill(-np.log(n_samples)) # equalize all weights

            # rejuvenate
            new_s0, new_lam1, new_lam2, new_lam3, new_V = prior.rvs(size=n_samples)

            new_evals = np.array([new_lam1,new_lam2,new_lam3]).T
            new_log_likelihoods = np.array([likelihood.logpdf(
                S0=new_s0[i], 
                evals=new_evals[i],
                evecs=new_V[i], 
                y=y
            ) for i in range(n_samples)], dtype=float)

            log_acceptance = beta * (new_log_likelihoods - log_likelihoods)
            
            accept = np.log(np.random.random(n_samples)) < log_acceptance

            s0[accept] = new_s0[accept]
            evals[accept] = new_evals[accept]
            V[accept] = new_V[accept]
            log_likelihoods[accept] = new_log_likelihoods[accept]

            #print(f"Rejuvenation acceptance: {np.mean(accept)}")
            states = np.column_stack((s0, evals, V.reshape(n_samples, -1)))

            #print(f"beta={beta:.4f}, ESS before={effective_sample_size:.0f}")
            #print("distinct before:", np.unique(
            #    np.column_stack((s0, evals, V.reshape(n_samples, -1))), axis=0
            #).shape[0])
            #print("distinct resampled indices:", np.unique(indices).size)
            #print("moves accepted:", np.count_nonzero(accept))

        prev_beta = beta
    order = np.argsort(evals, axis=1)
    evals = np.take_along_axis(evals, order, axis=1)
    V = np.take_along_axis(V, order[:, None, :], axis=2)
    return np.exp(log_weights), s0, evals, V



@disk_memoize()
def variational_inference(max_iters, K, learning_rate):
    # Students: implement Variational Inference here.
    # Before starting, make sure the prior, likelihood and variational_posterior are implemented.
    # Note: you may change, add, or remove input parameters depending on your design
    # (e.g. pass initialization values like those prepared in main()).

    raise NotImplementedError

    return variational_posterior(...)


@disk_memoize()
def laplace_approximation(y=None, gtab=None, point_estimate=None, prior=None, likelihood=None):
    """
    Computes the Laplace approximation of the posterior in transformed space theta in R^7.
    """
    # Load data and initial point estimate if not explicitly passed
    if y is None or point_estimate is None or gtab is None:
        y, point_estimate, gtab = get_preprocessed_data()

    S0_init, evals_init, evecs_init = point_estimate
    D_init = compute_D(evals_init, evecs_init).squeeze()

    # Initialize model components if not provided
    if prior is None:
        sigma = 29
        alpha_s = 2
        theta_s = 500
        alpha_lambda = 4
        theta_lambda = 2.5 * 10**(-4)
        prior = frozen_prior(alpha_s, theta_s, alpha_lambda, theta_lambda)

    if likelihood is None:
        sigma = 29
        likelihood = frozen_likelihood(gtab, sigma)

    # 1. Transform initial estimates into unconstrained space theta_0 in R^7
    theta_S0_init = np.log(S0_init)
    theta_D_init = theta_from_D(D_init)
    theta_init = np.concatenate([[theta_S0_init], theta_D_init])

    # 2. Define the objective function (Negative Log-Posterior)
    def neg_log_posterior(theta):
        S0 = np.exp(theta[0])
        D = D_from_theta(theta[1:])

        # Eigendecomposition to evaluate prior and likelihood
        evals, evecs = np.linalg.eigh(D)

        # Safeguard against negative/zero eigenvalues if numerical edge cases occur
        if np.any(evals <= 0):
            return 1e10

        log_p = prior.logpdf(S0, evals[0], evals[1], evals[2], evecs)
        log_l = likelihood.logpdf(S0, evecs, evals, y)

        total_log_post = log_l + log_p
        if np.isnan(total_log_post):
            return 1e10

        return -total_log_post

    # 3. Optimize to find the mode theta_hat
    opt_result = minimize(neg_log_posterior, theta_init, method='L-BFGS-B')
    theta_hat = opt_result.x

    # 4. Compute the Hessian matrix numerically at the mode using central differences
    def compute_hessian(f, x0, eps=1e-4):
        n = len(x0)
        hessian = np.zeros((n, n))
        f0 = f(x0)

        # Diagonal elements
        for i in range(n):
            x_plus, x_minus = x0.copy(), x0.copy()
            x_plus[i] += eps
            x_minus[i] -= eps
            hessian[i, i] = (f(x_plus) - 2 * f0 + f(x_minus)) / (eps**2)

        # Off-diagonal elements
        for i in range(n):
            for j in range(i + 1, n):
                x_pp, x_pm, x_mp, x_mm = x0.copy(), x0.copy(), x0.copy(), x0.copy()
                x_pp[i] += eps; x_pp[j] += eps
                x_pm[i] += eps; x_pm[j] -= eps
                x_mp[i] -= eps; x_mp[j] += eps
                x_mm[i] -= eps; x_mm[j] -= eps

                h_ij = (f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)) / (4 * eps**2)
                hessian[i, j] = h_ij
                hessian[j, i] = h_ij

        return hessian

    H = compute_hessian(neg_log_posterior, theta_hat)

    # 5. Invert Hessian to get Covariance Matrix Sigma = H^(-1)
    try:
        cov = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(H)

    # Ensure covariance matrix is strictly symmetric
    cov = 0.5 * (cov + cov.T)

    return mvn_reparameterized(mean=theta_hat, cov=cov)

# =============================================================================
# Posterior Uncertainty: 95% Credible Intervals 
# =============================================================================
def credible_intervals(S0, evals, evecs, evec_principal, method="", weights=None):    
    # 1. Compute scalar metrics from Laplace posterior samples
    S0_samples = S0.squeeze()
    md_samples = dti.mean_diffusivity(evals).squeeze()
    fa_samples = dti.fractional_anisotropy(evals).squeeze()
    angle_samples = (360 / (2 * np.pi)) * np.arccos(np.abs(np.dot(evecs[:, :, 2], evec_principal)))

    # 2. Calculate 2.5th and 97.5th percentiles (95% CI)
    if weights is None:
        ci_S0 = np.percentile(S0_samples, [2.5, 97.5])
        ci_md = np.percentile(md_samples, [2.5, 97.5])
        ci_fa = np.percentile(fa_samples, [2.5, 97.5])
        ci_angle = np.percentile(angle_samples, [2.5, 97.5])
    else:
        def interval(samples):
            return DescrStatsW(data=np.asarray(samples).reshape(-1),
                                weights=np.asarray(weights).reshape(-1)
            ).quantile(probs=[0.025, 0.975], return_pandas=False)
        
        ci_S0 = interval(S0_samples)
        ci_md = interval(md_samples)
        ci_fa = interval(fa_samples)
        ci_angle = interval(angle_samples)

    # 3. Print formatted results
    print(f"\n95% credible intervals of psterior uncertainty for method '{method}'")
    print(f"S0 95% CI: [{ci_S0[0]:.2f}, {ci_S0[1]:.2f}]")
    print(f"Mean Diffusivity 95% CI: [{ci_md[0]:.6f}, {ci_md[1]:.6f}]")
    print(f"Fractional Anisotropy 95% CI: [{ci_fa[0]:.3f}, {ci_fa[1]:.3f}]")
    print(f"Acute Angle 95% CI: [{ci_angle[0]:.2f}°, {ci_angle[1]:.2f}°]\n")
    





"""
=============================================================================
Visualization & Experiment Runner
=============================================================================
Plotting function and the main() script to run experiments.
"""

def main():

    # Initialize with preprocessed data and DTI point estimate
    # (these values can be used as starting points for inference methods)

    y, point_estimate, gtab = get_preprocessed_data(force_recompute=False)
    S0_init, evals_init, evecs_init = point_estimate
    D_init = compute_D(evals_init, evecs_init).squeeze()

    # Find principal eigenvector from DTI estimate (for plotting)
    evec_principal = evecs_init[:, 0]
    # Set random seed and number of posterior samples
    np.random.seed(0)
    n_samples = 10000

    #hyperparameters:
    sigma = 29
    alpha_s = 2
    theta_s = 500
    alpha_lambda = 4
    theta_lambda = 2.5*10**(-4)

    prior = frozen_prior(alpha_s, theta_s, alpha_lambda, theta_lambda)


    #print("prior logpdf: ",prior.logpdf(S0_init,evals_init[0],evals_init[1],evals_init[2],evecs_init))

    likelihood = frozen_likelihood(gtab, sigma)
    a = likelihood.logpdf(S0_init, evecs_init, evals_init, y)
    #print("likelihood logpdf: ",a)
    
    # Run Metropolis–Hastings and plot results
    gamma_param = 0.98
    nu_param = 105
    S0_mh, evals_mh, evecs_mh = metropolis_hastings(y, n_samples, prior, likelihood,
                                                    gamma_param, nu_param,
                                                    S0_init, D_init, plot_traces=False,
                                                    force_recompute=False)
    burn_in = 0
    plot_results(S0_mh[burn_in:], evals_mh[burn_in:], evecs_mh[burn_in:, :, :], evec_principal, method="mh")
    credible_intervals(S0_mh[burn_in:], evals_mh[burn_in:], evecs_mh[burn_in:, :, :], evec_principal, method="Metropolis-Hastings")

    print("Done with MH.")


    # Run Importance Sampling and plot results
    # gamma_param, nu_param = 1, 10
    #gamma_params = [0.9, 0.93, 0.94, 0.95, 0.96,0.97,0.98,0.99, 1, 1.05, 1.1, 1.2]
    #nu_params = [100, 101, 102,103,104, 105,110]
    #for gamma_param in gamma_params:
    #    for nu_param in nu_params: 
    #        #a = likelihood.logpdf(S0_init, evecs_init, evals_init, y)
    #        w_is, S0_is, evals_is, evecs_is = importance_sampling(
    #            prior, likelihood, y, n_samples, gamma_param, nu_param, S0_init, D_init
    #        )#, force_recompute=False)
    #        normalized_importance_weights = np.exp(w_is - logsumexp(w_is))
    #        effective_sample_size = 1/np.sum(normalized_importance_weights**2)
    #        print(f"effective sample size (N_ESS) at gamma={gamma_param} and nu={nu_param}: {effective_sample_size}")
    # manual testing gives these as optimal (with roughly 20 effective samples, which is quite bad)
    gamma_param = 0.98
    nu_param=105
    w_is, S0_is, evals_is, evecs_is = importance_sampling(
        prior, likelihood, y, n_samples, gamma_param, nu_param, S0_init, D_init
    )#, force_recompute=False)
    normalized_importance_weights = np.exp(w_is - logsumexp(w_is))
    effective_sample_size = 1/np.sum(normalized_importance_weights**2)
    print(f"effective sample size (N_ESS) at gamma={gamma_param} and nu={nu_param}: {effective_sample_size} out of total {n_samples}")
    
    plot_results(S0_is, evals_is, evecs_is, evec_principal, weights=normalized_importance_weights, method="is", large_text=True)
    
    credible_intervals(S0_is, evals_is, evecs_is, evec_principal, method="Importance sampling", weights=normalized_importance_weights)
    w_smc, S0_smc, evals_smc, evecs_smc = sequential_monte_carlo_sampling(
        prior, likelihood, y, n_samples, gamma_param, nu_param, S0_init, D_init
    )

    plot_results(S0_smc, evals_smc, evecs_smc, evec_principal, weights=w_smc, method="smc", large_text=True)
    credible_intervals(S0_smc, evals_smc, evecs_smc, evec_principal, method="Sequential Monte Carlo sampling")

    # Run Variational Inference and plot results
    # posterior_vi = variational_inference(force_recompute=False)
    # S0_vi, evals_vi, evecs_vi = posterior_vi.rvs(size=n_samples)
    # plot_results(S0_vi, evals_vi, evecs_vi, evec_principal, method="vi")
    #credible_intervals(S0_vi, evals_vi, evecs_vi, evec_principal, method="Variational inference")

    # # Run Laplace Approximation and plot results
    #posterior_laplace = laplace_approximation(force_recompute=False)
    #S0_laplace, evals_laplace, evecs_laplace = posterior_laplace.rvs(size=n_samples)
    #plot_results(S0_laplace, evals_laplace, evecs_laplace, evec_principal, method="laplace")
    #credible_intervals(S0_laplace, evals_laplace, evecs_laplace, evec_principal, method="Laplace Approximation")

    print("Done.")


def plot_results(S0, evals, evecs, evec_ref, weights=None, method="", large_text=False):
    """
    Plot posterior results as histograms and save to file.

    Creates histograms of baseline signal (S0), mean diffusivity (MD),
    fractional anisotropy (FA), and the angle between estimated and
    reference eigenvectors.

    Parameters
    ----------
    S0 : ndarray
        Sampled baseline signals.
    evals : ndarray
        Sampled eigenvalues of the diffusion tensor.
    evecs : ndarray
        Sampled eigenvectors of the diffusion tensor.
    evec_ref : ndarray
        Reference principal eigenvector (from point estimate).
    weights : ndarray, optional
        Importance weights for samples. Uniform if None.
    method : str
        Name of inference method (used in output filename).
    """
    
    # Use uniform weights if none provided
    if weights is None:
        weights = np.ones_like(S0)
        weights /= np.sum(weights)

    # Choose number of bins based on sample size
    n_bins = np.floor(np.sqrt(len(weights))).astype(int)

    # Squeeze arrays for plotting
    weights = weights.squeeze()
    S0 = S0.squeeze()
    md = dti.mean_diffusivity(evals).squeeze()
    fa = dti.fractional_anisotropy(evals).squeeze()

    # Compute acute angle between estimated and reference eigenvectors
    angle = 360/(2*np.pi) * np.arccos(np.abs(np.dot(evecs[:, :, 2], evec_ref)))
    
    # Create 2x2 grid of histograms
    fig, axes = plt.subplots(2, 2, figsize=(12, 12), sharey=False)
    if(large_text):
        for ax in axes.flat:
            ax.tick_params(axis="both", labelsize=16)
            ax.xaxis.label.set_size(20)
            ax.yaxis.label.set_size(20)
            ax.xaxis.get_offset_text().set_fontsize(16)
            ax.yaxis.get_offset_text().set_fontsize(16)


    axes[0, 0].hist(S0, bins=n_bins, density=True, weights=weights, 
                    alpha=0.7, color='red', edgecolor='black')
    axes[0, 0].set_xlabel("S0")
    axes[0, 0].set_ylabel("Density")

    axes[0, 1].hist(md, bins=n_bins, density=True, weights=weights, 
                    alpha=0.7, color='green', edgecolor='black')
    axes[0, 1].set_xlabel("Mean diffusivity")
    axes[0, 1].set_ylabel("Density")
    if(large_text): # fix unreadable text
        axes[0, 1].ticklabel_format(
            axis="x", style="sci", scilimits=(0,0), useMathText=True, useOffset=False
        )
    axes[1, 0].hist(fa, bins=n_bins, density=True, weights=weights,
                     alpha=0.7, color='blue', edgecolor='black')
    axes[1, 0].set_xlabel("Fractional anisotropy")
    axes[1, 0].set_ylabel("Density")

    axes[1, 1].hist(angle, bins=n_bins, density=True, weights=weights, 
                    alpha=0.7, color='magenta', edgecolor='black')
    axes[1, 1].set_xlabel("Acute angle")
    axes[1, 1].set_ylabel("Density")

    # Adjust layout and save figure with method name
    plt.tight_layout()
    plt.savefig("results_{}.png".format(method), dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    main()
