import numpy as np
from scipy.stats import wishart


class MultivariateNormal:
    def __init__(self, d):
        self.d = d
        self.A = wishart.rvs(df=d, scale=np.eye(d)) # precision matrix

    def log_p(self, theta):
        return -0.5 * np.dot(theta, np.dot(self.A, theta))
    
    def grad_log_p(self, theta):
        return -np.dot(self.A, theta)
    