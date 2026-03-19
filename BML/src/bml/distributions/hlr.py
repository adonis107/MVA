import numpy as np
from scipy.special import expit


class HierarchicalLogisticRegression:
    def __init__(self, X, y, N, d_coeffs, lam):
        self.X = X
        self.y = y
        self.N = N
        self.d_coeffs = d_coeffs
        self.lam = lam

    def log_p(self, theta):
        """
        Evaluates the log-posterior for Hierarchical Bayesian Logistic Regression.
        theta[:301] = alpha and beta coefficients
        theta[301] = v = log(sigma^2)
        """
        if np.any(np.abs(theta) > 500.0) or theta[-1] < -20.0 or theta[-1] > 20.0:
            return -np.inf

        coeffs = theta[:self.d_coeffs]
        v = theta[-1]

        sigma_sq = np.exp(v)
        inv_sigma_sq = np.exp(-v)  # 1/sigma^2 for numerical stability

        # Log-likelihood
        z = -self.y * np.dot(self.X, coeffs)
        log_likelihood = -np.sum(np.logaddexp(0, z))

        # Log-prior
        log_prior = -0.5 * (np.sum(coeffs**2) * inv_sigma_sq) - (self.d_coeffs / 2) * v - self.lam * sigma_sq + v

        return log_likelihood + log_prior

    def grad_log_p(self, theta):
        """
        Evaluates the gradient of the log-posterior for HLR.
        """
        theta_safe = np.clip(theta, -1000.0, 1000.0)
        theta_safe[-1] = np.clip(theta_safe[-1], -20.0, 20.0)

        coeffs = theta_safe[:self.d_coeffs]
        v = theta_safe[-1]

        sigma_sq = np.exp(v)
        inv_sigma_sq = np.exp(-v)  # 1/sigma^2 for numerical stability
        grad = np.zeros_like(theta)

        # Gradient
        z = -self.y * np.dot(self.X, coeffs)
        s = expit(z)
        grad_likelihood_coeffs = np.dot(self.X.T, self.y * s)
        grad_prior_coeffs = -coeffs * inv_sigma_sq

        grad[:self.d_coeffs] = grad_likelihood_coeffs + grad_prior_coeffs

        # Gradient w.r.t. v = log(sigma^2)
        grad[-1] = 0.5 * np.sum(coeffs**2) * inv_sigma_sq - (self.d_coeffs / 2.0) - self.lam * sigma_sq + 1.0

        return grad
    