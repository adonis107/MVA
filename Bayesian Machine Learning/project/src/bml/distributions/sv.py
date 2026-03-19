import numpy as np
from scipy.special import digamma
from scipy.stats import t


class StochasticVolatility:
    def __init__(self, R):
        self.R = R

    def log_p(self, theta):
        # safety
        if np.any(np.abs(theta[:-1]) > 30.0) or theta[-1] < -15.0 or theta[-1] > 15.0:
            return -np.inf

        x = theta[:-1]   
        alpha = theta[-1]

        v = np.exp(alpha)
        s = np.exp(x)
        z = self.R * np.exp(-x)

        # Exponential priors on nu and s_1
        L1 = -0.01 * v - 0.01 * s[0]

        # Student-t likelihoods
        L2 = np.sum(t.logpdf(z, df=v))

        # Integrated precision parameter tau
        S_sq = np.sum((x[1:] - x[:-1])**2)
        U = 0.01 + 0.5 * S_sq
        L3 = -(3001.0 / 2.0) * np.log(U)

        # Log-Jacobian of transformations
        L4 = alpha + np.sum(x)

        return L1 + L2 + L3 + L4

    def grad_log_p(self, theta):
        theta_safe = np.clip(theta, -30.0, 30.0)
        theta_safe[-1] = np.clip(theta[-1], -15.0, 15.0)

        x = theta_safe[:-1]
        alpha = theta_safe[-1]

        v = np.exp(alpha)
        s = np.exp(x)
        z = self.R * np.exp(-x)

        z_sq = z**2
        v_plus_z_sq = v + z_sq
        v_plus_1 = v + 1.0

        # x gradients
        dL1_dx = np.zeros_like(x)
        dL1_dx[0] = -0.01 * s[0]

        dL2_dx = (v_plus_1 * z_sq) / v_plus_z_sq

        diff_x = x[1:] - x[:-1]
        dL3_dx = np.zeros_like(x)
        coef = -(3001.0 / (2.0 * (0.01 + 0.5 * np.sum(diff_x**2))))

        dL3_dx[0] = coef * (-diff_x[0])
        dL3_dx[1:-1] = coef * (diff_x[:-1] - diff_x[1:])
        dL3_dx[-1] = coef * diff_x[-1]

        dL4_dx = np.ones_like(x)

        dx = dL1_dx + dL2_dx + dL3_dx + dL4_dx

        # alpha gradient
        dL1_dalpha = -0.01 * v

        digamma_half_v_plus_1 = digamma((v + 1.0) / 2.0)
        digamma_half_v = digamma(v / 2.0)

        dL2_dv_terms = (0.5 * digamma_half_v_plus_1 - 0.5 * digamma_half_v - 0.5 / v 
                        - 0.5 * np.log(1.0 + z_sq / v) 
                        + (v_plus_1 * z_sq) / (2.0 * v * v_plus_z_sq))

        dL2_dalpha = v * np.sum(dL2_dv_terms)
        dL4_dalpha = 1.0

        dalpha = dL1_dalpha + dL2_dalpha + dL4_dalpha

        return np.append(dx, dalpha)
    