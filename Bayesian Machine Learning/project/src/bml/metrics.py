import numpy as np

def compute_autocorrelation(samples, s, mu, var):
    """Computes the autocorrelation at lag s."""
    M = len(samples)
    autocovariance = np.sum((samples[s:] - mu) * (samples[:-s] - mu)) / (M - s)
    return autocovariance / var

def compute_ess_1d(samples, mu, var):
    """Computes the ESS for a single 1D array of samples based on Appendix A."""
    M = len(samples)
    autocorrelations = []
    
    for s in range(1, M):
        rho_s = compute_autocorrelation(samples, s, mu, var)
        if rho_s < 0.05:
            break
        autocorrelations.append((1 - s / M) * rho_s)
        
    # ESS estimator
    ess = M / (1 + 2 * sum(autocorrelations))
    return ess

def evaluate_mvn_efficiency(samples, A):
    """
    Evaluates the worst-case ESS across all dimensions for both the mean 
    and the second central moment, as described in Section 4.
    """
    M, D = samples.shape
    
    true_cov = np.linalg.inv(A)
    true_variances = np.diag(true_cov)
    true_means = np.zeros(D)
    
    min_ess = float('inf')
    
    for d in range(D):
        dim_samples = samples[:, d]
        
        # ESS for the mean
        mu = true_means[d]
        var = true_variances[d]
        ess_mean = compute_ess_1d(dim_samples, mu, var)
        
        # ESS for the variance
        moment_samples = (dim_samples - mu)**2
        moment_mu = np.mean(moment_samples)
        moment_var = np.var(moment_samples)
        ess_variance = compute_ess_1d(moment_samples, moment_mu, moment_var)
        
        # Take the worst-case ESS for this dimension
        dim_min_ess = min(ess_mean, ess_variance)
        
        if dim_min_ess < min_ess:
            min_ess = dim_min_ess
            
    return min_ess