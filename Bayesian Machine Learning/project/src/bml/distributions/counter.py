class GradCounter:
    """Wrapper to count gradient evaluations."""
    def __init__(self, distribution):
        self.distribution = distribution
        self.count = 0

    def grad(self, theta):
        self.count += 1
        return self.distribution.grad_log_p(theta)

    def log_p(self, theta):
        return self.distribution.log_p(theta)