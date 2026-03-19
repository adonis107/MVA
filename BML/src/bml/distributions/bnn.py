import numpy as np


def _sigmoid(z):
    """Numerically stable sigmoid."""
    pos = z >= 0
    neg = ~pos
    result = np.empty_like(z, dtype=np.float64)
    result[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[neg])
    result[neg] = ez / (1.0 + ez)
    return result


class BayesianNeuralNetwork:
    """
    A 2-hidden-layer Bayesian Neural Network for binary classification.

    Architecture: input_dim -> h1 -> h2 -> 1
    Activation: tanh (hidden layers), sigmoid (output)
    Prior: Normal(0, sigma^2) on all weights and biases.
    """

    def __init__(self, X, y, hidden_dims=(10, 10), sigma_sq=100.0):
        self.X = X
        self.y = y
        self.N, self.input_dim = X.shape
        self.h1, self.h2 = hidden_dims
        self.sigma_sq = sigma_sq

        self.n_W1 = self.input_dim * self.h1
        self.n_b1 = self.h1
        self.n_W2 = self.h1 * self.h2
        self.n_b2 = self.h2
        self.n_W3 = self.h2
        self.n_b3 = 1

        self.d = self.n_W1 + self.n_b1 + self.n_W2 + self.n_b2 + self.n_W3 + self.n_b3

        # slice boundaries
        idx = 0
        self.s_W1 = (idx, idx + self.n_W1); idx += self.n_W1
        self.s_b1 = (idx, idx + self.n_b1); idx += self.n_b1
        self.s_W2 = (idx, idx + self.n_W2); idx += self.n_W2
        self.s_b2 = (idx, idx + self.n_b2); idx += self.n_b2
        self.s_W3 = (idx, idx + self.n_W3); idx += self.n_W3
        self.s_b3 = (idx, idx + self.n_b3)

    def _unpack(self, theta):
        """Unpack flat parameter vector into weight matrices and biases."""
        W1 = theta[self.s_W1[0]:self.s_W1[1]].reshape(self.input_dim, self.h1)
        b1 = theta[self.s_b1[0]:self.s_b1[1]]
        W2 = theta[self.s_W2[0]:self.s_W2[1]].reshape(self.h1, self.h2)
        b2 = theta[self.s_b2[0]:self.s_b2[1]]
        W3 = theta[self.s_W3[0]:self.s_W3[1]].reshape(self.h2, 1)
        b3 = theta[self.s_b3[0]:self.s_b3[1]]
        return W1, b1, W2, b2, W3, b3

    def _forward(self, theta):
        """Forward pass returning all intermediate activations."""
        W1, b1, W2, b2, W3, b3 = self._unpack(theta)

        # Layer 1
        z1 = self.X @ W1 + b1
        a1 = np.tanh(z1)

        # Layer 2
        z2 = a1 @ W2 + b2
        a2 = np.tanh(z2)

        # Output layer
        z3 = (a2 @ W3 + b3).ravel()
        p = _sigmoid(z3) 

        return W1, b1, W2, b2, W3, b3, z1, a1, z2, a2, z3, p

    def log_p(self, theta):
        """Log-posterior = log-likelihood + log-prior."""
        if np.any(np.abs(theta) > 500.0):
            return -np.inf

        _, _, _, _, _, _, _, _, _, _, z3, p = self._forward(theta)

        # Numerically stable binary cross-entropy
        log_lik = np.sum(
            self.y * (-np.logaddexp(0, -z3)) + (1 - self.y) * (-np.logaddexp(0, z3))
        )

        # Log-prior: Normal(0, sigma^2)
        log_prior = -0.5 * np.sum(theta ** 2) / self.sigma_sq

        return log_lik + log_prior

    def grad_log_p(self, theta):
        """Gradient of the log-posterior w.r.t. theta."""
        theta_safe = np.clip(theta, -500.0, 500.0)
        W1, b1, W2, b2, W3, b3, z1, a1, z2, a2, z3, p = self._forward(theta_safe)

        # Backprop through likelihood
        delta3 = (self.y - p).reshape(-1, 1)

        # Output layer gradients
        grad_W3 = a2.T @ delta3
        grad_b3 = np.sum(delta3, axis=0)

        # Backprop to layer 2
        delta2 = (delta3 @ W3.T) * (1 - a2 ** 2)
        grad_W2 = a1.T @ delta2
        grad_b2 = np.sum(delta2, axis=0)

        # Backprop to layer 1
        delta1 = (delta2 @ W2.T) * (1 - a1 ** 2)
        grad_W1 = self.X.T @ delta1
        grad_b1 = np.sum(delta1, axis=0)

        # Gradient of log-prior
        grad_prior = -theta_safe / self.sigma_sq

        # Pack gradients
        grad = np.zeros_like(theta_safe)
        grad[self.s_W1[0]:self.s_W1[1]] = grad_W1.ravel()
        grad[self.s_b1[0]:self.s_b1[1]] = grad_b1
        grad[self.s_W2[0]:self.s_W2[1]] = grad_W2.ravel()
        grad[self.s_b2[0]:self.s_b2[1]] = grad_b2
        grad[self.s_W3[0]:self.s_W3[1]] = grad_W3.ravel()
        grad[self.s_b3[0]:self.s_b3[1]] = grad_b3

        grad += grad_prior

        return grad
