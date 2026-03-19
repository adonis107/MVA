import numpy as np


class NealsFunnel:
    """
    Neal's Funnel distribution (Neal, 2003).

    A 10-dimensional hierarchical model:
        v   ~ N(0, sigma_v^2)          [default sigma_v = 3]
        x_i ~ N(0, exp(v))             for i = 1, ..., 9

    The parameter vector theta = [v, x_1, ..., x_9] has dimension d = 10.

    The log-density is:
        log p(v, x) = -v^2 / (2 * sigma_v^2)
                      - 9/2 * v
                      - 1/2 * exp(-v) * sum(x_i^2)

    As v -> -inf the conditional variance exp(v) -> 0, creating the
    narrow funnel neck that makes sampling extremely challenging.
    """

    def __init__(self, d=10, sigma_v=3.0):
        self.d = d
        self.sigma_v = sigma_v
        self.sigma_v_sq = sigma_v ** 2

    def log_p(self, theta):
        """Log-density of Neal's Funnel."""
        v = theta[0]
        x = theta[1:]

        # Overflow safety
        if np.abs(v) > 50.0 or np.any(np.abs(x) > 1e6):
            return -np.inf

        log_pv = -0.5 * v ** 2 / self.sigma_v_sq
        log_px_given_v = -0.5 * (self.d - 1) * v - 0.5 * np.exp(-v) * np.sum(x ** 2)

        return log_pv + log_px_given_v

    def grad_log_p(self, theta):
        """Gradient of the log-density w.r.t. theta."""
        theta_safe = theta.copy()
        theta_safe[0] = np.clip(theta_safe[0], -50.0, 50.0)

        v = theta_safe[0]
        x = theta_safe[1:]

        grad = np.zeros_like(theta_safe)

        sum_x_sq = np.sum(x ** 2)
        grad[0] = -v / self.sigma_v_sq - 0.5 * (self.d - 1) + 0.5 * np.exp(-v) * sum_x_sq
        grad[1:] = -x * np.exp(-v)

        return grad
