import numpy as np
from scipy.special import expit


class LogisticRegression:
    def __init__(self, X, y, sigma_sq):
        self.X = X
        self.y = y
        self.sigma_sq = sigma_sq

    def log_p(self, theta):
        """
        Evaluates the log-posterior for Bayesian Logistic Regression.
        theta[0] represents alpha, theta[1:] represents beta.
        """
        # Prior
        log_prior = -0.5 * np.sum(theta**2) / self.sigma_sq

        # Likelihood
        z = -self.y * np.dot(self.X, theta)
        log_likelihood = -np.sum(np.logaddexp(0, z))

        return log_prior + log_likelihood

    def grad_log_p(self, theta):
        """
        Evaluates the gradient of the log-posterior.
        """
        # Gradient of prior
        grad_prior = -theta / self.sigma_sq

        # Gradient of likelihood
        z = -self.y * np.dot(self.X, theta)
        s = expit(z)
        grad_likelihood = np.dot(self.X.T, self.y * s)

        return grad_prior + grad_likelihood
