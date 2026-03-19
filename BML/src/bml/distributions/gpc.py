import numpy as np
from scipy.special import expit


class GPClassification:
    """
    Gaussian Process Classification with an RBF (Squared Exponential) kernel.

    The latent function values f at training points follow a GP prior:
        f ~ N(0, K)
    where K_ij = signal_var * exp(-||x_i - x_j||^2 / (2 * length_scale^2)) + jitter * I

    The likelihood is Bernoulli with a logistic link:
        p(y_i = 1 | f_i) = sigmoid(f_i)

    The parameters being sampled are the N latent function values f.
    This creates a fully dense N x N covariance matrix, testing the
    sampler's ability to handle intense parameter correlations.
    """

    def __init__(self, X, y, length_scale=1.0, signal_var=1.0, jitter=1e-6):
        self.X = X
        self.y = y
        self.N = X.shape[0]
        self.d = self.N

        self.length_scale = length_scale
        self.signal_var = signal_var
        self.jitter = jitter

        # Kernel matrix and inverse / Cholesky
        self.K = self._rbf_kernel(X)
        self.L_chol = np.linalg.cholesky(self.K)
        self.K_inv = np.linalg.solve(
            self.K, np.eye(self.N)
        )

        # Log-determinant for normalisation
        self.log_det_K = 2.0 * np.sum(np.log(np.diag(self.L_chol)))

    def _rbf_kernel(self, X):
        """Compute the RBF (Squared Exponential) kernel matrix."""
        sq_dists = np.sum(X ** 2, axis=1, keepdims=True) \
                   - 2.0 * X @ X.T \
                   + np.sum(X ** 2, axis=1, keepdims=True).T
        sq_dists = np.maximum(sq_dists, 0.0)  # numerical safety
        K = self.signal_var * np.exp(-0.5 * sq_dists / self.length_scale ** 2)
        K += self.jitter * np.eye(self.N)
        return K

    def log_p(self, f):
        """
        Log-posterior of the latent function values f.

        log p(f | y, X) ∝ log p(y | f) + log p(f)
        = -sum_i log(1 + exp(-y_i * f_i)) - 0.5 * f^T K^{-1} f
        """
        if np.any(np.abs(f) > 500.0):
            return -np.inf

        # Log-prior
        log_prior = -0.5 * f @ self.K_inv @ f

        # Log-likelihood
        z = -self.y * f
        log_lik = -np.sum(np.logaddexp(0, z))

        return log_prior + log_lik

    def grad_log_p(self, f):
        """
        Gradient of the log-posterior w.r.t. f.

        d/df log p(f | y, X) = -K^{-1} f + y * sigmoid(-y * f)
        """
        f_safe = np.clip(f, -500.0, 500.0)

        # Gradient of log-prior
        grad_prior = -self.K_inv @ f_safe

        # Gradient of log-likelihood
        z = -self.y * f_safe
        grad_lik = self.y * expit(z)

        return grad_prior + grad_lik
