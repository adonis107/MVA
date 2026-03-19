import numpy as np
from bml.samplers.utils import leapfrog, find_reasonable_epsilon


class NaiveNUTS():
    def __init__(self, epsilon, L, grad, max_depth=10):
        self.epsilon = epsilon
        self.L = L
        self.grad = grad
        self.max_depth = max_depth

    def build_tree(self, theta, r, log_u, v, j, epsilon, L, grad, delta_max=1000):
        """Recursively builds a binary tree for the NUTS algorithm."""
        if j == 0:
            # Base case: take a single leapfrog step in the direction v
            theta_prime, r_prime = leapfrog(theta, r, v * epsilon, grad)

            # Check if the new state is valid
            log_prob = L(theta_prime) - 0.5 * np.dot(r_prime, r_prime)
            if log_u <= log_prob:
                C_prime = [(theta_prime, r_prime)]
            else:
                C_prime = []

            # indicator for stopping criterion being met
            s_prime = int(log_prob > log_u - delta_max)

            return theta_prime, r_prime, theta_prime, r_prime, C_prime, s_prime

        else:
            # Recursion: build the left and right subtrees
            theta_minus, r_minus, theta_plus, r_plus, C_prime, s_prime = self.build_tree(theta, r, log_u, v, j-1, epsilon, L, grad, delta_max)

            if v == -1:
                theta_minus, r_minus, _, _, C_double_prime, s_double_prime = self.build_tree(theta_minus, r_minus, log_u, v, j-1, epsilon, L, grad, delta_max)
            else:
                _, _, theta_plus, r_plus, C_double_prime, s_double_prime = self.build_tree(theta_plus, r_plus, log_u, v, j-1, epsilon, L, grad, delta_max)

            s_prime = s_prime and s_double_prime and int(np.dot(theta_plus - theta_minus, r_minus) >= 0) and int(np.dot(theta_plus - theta_minus, r_plus) >= 0)
            C_prime = C_prime + C_double_prime

            return theta_minus, r_minus, theta_plus, r_plus, C_prime, s_prime

    def sample(self, theta0, M):
        """Samples from the target distribution using the Naive NUTS algorithm."""
        theta = theta0
        samples = [theta0]
        for m in range(1, M+1):
            # Sample momentum
            r0 = np.random.normal(size=theta0.shape)

            # Sample slice variable
            joint_prob = self.L(theta) - 0.5 * np.dot(r0, r0)
            log_u = joint_prob - np.random.exponential(1)

            # Initialize tree
            theta_minus = theta
            theta_plus = theta
            r_minus = r0
            r_plus = r0
            j = 0
            C = [(theta, r0)]
            s = 1

            # Build the tree until the stopping criterion is met
            while s == 1 and j < self.max_depth:
                # Choose a direction
                v = np.random.choice([-1, 1])

                if v == -1:
                    theta_minus, r_minus, _, _, C_prime, s_prime = self.build_tree(theta_minus, r_minus, log_u, v, j, self.epsilon, self.L, self.grad)
                else:
                    _, _, theta_plus, r_plus, C_prime, s_prime = self.build_tree(theta_plus, r_plus, log_u, v, j, self.epsilon, self.L, self.grad)

                if s_prime == 1:
                    C = C + C_prime

                s = s_prime and int(np.dot(theta_plus - theta_minus, r_minus) >= 0) and int(np.dot(theta_plus - theta_minus, r_plus) >= 0)
                j += 1

            # Sample theta_m, r uniformly at random from C
            theta, r = list(C)[np.random.choice(len(C))]
            samples.append(theta)
        return np.array(samples)
    

class EfficientNUTS():
    def __init__(self, epsilon, L, grad, max_depth=10):
        self.epsilon = epsilon
        self.L = L
        self.grad = grad
        self.max_depth = max_depth

    def build_tree(self, theta, r, log_u, v, j, epsilon, L, grad, delta_max=1000):
        """Recursively builds a binary tree for the Efficient NUTS algorithm."""
        if j == 0:
            # Base case: take a single leapfrog step in the direction v
            theta_prime, r_prime = leapfrog(theta, r, v * epsilon, grad)

            log_prob = L(theta_prime) - 0.5 * np.dot(r_prime, r_prime)
            n_prime = int(log_u <= log_prob)
            s_prime = int(log_prob > log_u - delta_max)

            return theta_prime, r_prime, theta_prime, r_prime, theta_prime, n_prime, s_prime
        
        else:
            # Recursion: build the left and right subtrees
            theta_minus, r_minus, theta_plus, r_plus, theta_prime, n_prime, s_prime = self.build_tree(theta, r, log_u, v, j-1, epsilon, L, grad, delta_max)

            if s_prime == 1:
                if v == -1:
                    theta_minus, r_minus, _, _, theta_double_prime, n_double_prime, s_double_prime = self.build_tree(theta_minus, r_minus, log_u, v, j-1, epsilon, L, grad, delta_max)
                else:
                    _, _, theta_plus, r_plus, theta_double_prime, n_double_prime, s_double_prime = self.build_tree(theta_plus, r_plus, log_u, v, j-1, epsilon, L, grad, delta_max)

                if np.random.uniform() < n_double_prime / (n_prime + n_double_prime + 1e-10):
                    theta_prime = theta_double_prime

                s_prime = s_double_prime and int(np.dot(theta_plus - theta_minus, r_minus) >= 0) and int(np.dot(theta_plus - theta_minus, r_plus) >= 0)
                n_prime += n_double_prime

            return theta_minus, r_minus, theta_plus, r_plus, theta_prime, n_prime, s_prime

    def sample(self, theta0, M):
        """Samples from the target distribution using the Efficient NUTS algorithm."""
        theta_prev = theta0
        samples = [theta0]

        for m in range(1, M+1):
            # Sample momentum
            r0 = np.random.normal(size=theta_prev.shape)

            # Sample slice variable
            joint_prob = self.L(theta_prev) - 0.5 * np.dot(r0, r0)
            log_u = joint_prob - np.random.exponential(1)

            # Initialize tree
            theta_minus = theta_prev
            theta_plus = theta_prev
            r_minus = r0
            r_plus = r0
            j = 0
            theta_next = theta_prev
            n = 1
            s = 1

            # Build the tree until the stopping criterion is met
            while s == 1 and j < self.max_depth:
                # Choose a direction
                v = np.random.choice([-1, 1])

                if v == -1:
                    theta_minus, r_minus, _, _, theta_prime, n_prime, s_prime = self.build_tree(theta_minus, r_minus, log_u, v, j, self.epsilon, self.L, self.grad)
                else:
                    _, _, theta_plus, r_plus, theta_prime, n_prime, s_prime = self.build_tree(theta_plus, r_plus, log_u, v, j, self.epsilon, self.L, self.grad)

                if s_prime == 1 and np.random.uniform() < min(1, n_prime / (n + 1e-10)):
                        theta_next = theta_prime
                
                n += n_prime
                s = s_prime and int(np.dot(theta_plus - theta_minus, r_minus) >= 0) and int(np.dot(theta_plus - theta_minus, r_plus) >= 0)
                j += 1

            theta_prev = theta_next
            samples.append(theta_prev)
            
        return np.array(samples)


class DualAveragingNUTS():
    def __init__(self, L, grad, max_depth=10):
        self.L = L
        self.grad = grad
        self.max_depth = max_depth

    def build_tree(self, theta, r, log_u, v, j, epsilon, theta0, r0, L, grad, delta_max=1000):
        """Recursively builds a binary tree for the Dual Averaging NUTS algorithm."""
        if j == 0:
            # Base case: take a single leapfrog step in the direction v
            theta_prime, r_prime = leapfrog(theta, r, v * epsilon, grad)

            joint_prob = L(theta_prime) - 0.5 * np.dot(r_prime, r_prime)

            n_prime = int(log_u <= joint_prob)
            s_prime = int(log_u <= delta_max + joint_prob)

            joint_prob0 = L(theta0) - 0.5 * np.dot(r0, r0)
            alpha = min(1, np.exp(joint_prob - joint_prob0))

            return theta_prime, r_prime, theta_prime, r_prime, theta_prime, n_prime, s_prime, alpha, 1
        
        else:
            # Recursion: implicitly build the left and right subtrees
            theta_minus, r_minus, theta_plus, r_plus, theta_prime, n_prime, s_prime, alpha_prime, n_alpha_prime = self.build_tree(theta, r, log_u, v, j-1, epsilon, theta0, r0, L, grad, delta_max)

            if s_prime == 1:
                if v == -1:
                    theta_minus, r_minus, _, _, theta_double_prime, n_double_prime, s_double_prime, alpha_double_prime, n_alpha_double_prime = self.build_tree(theta_minus, r_minus, log_u, v, j-1, epsilon, theta0, r0, L, grad, delta_max)
                else:
                    _, _, theta_plus, r_plus, theta_double_prime, n_double_prime, s_double_prime, alpha_double_prime, n_alpha_double_prime = self.build_tree(theta_plus, r_plus, log_u, v, j-1, epsilon, theta0, r0, L, grad, delta_max)

                if np.random.uniform() < n_double_prime / (n_prime + n_double_prime + 1e-10):
                    theta_prime = theta_double_prime

                alpha_prime += alpha_double_prime
                n_alpha_prime += n_alpha_double_prime
                s_prime = s_double_prime and int(np.dot(theta_plus - theta_minus, r_minus) >= 0) and int(np.dot(theta_plus - theta_minus, r_plus) >= 0)
                n_prime += n_double_prime

            return theta_minus, r_minus, theta_plus, r_plus, theta_prime, n_prime, s_prime, alpha_prime, n_alpha_prime

    def sample(self, theta0, delta, M, M_adapt):
        """Samples from the target distribution using the Dual Averaging NUTS algorithm."""
        # Dual averaging parameters
        epsilon = find_reasonable_epsilon(theta0, self.grad, self.L)
        mu = np.log(10 * epsilon)
        epsilon_bar  = 1.0
        H_bar = 0.0
        gamma = 0.05
        t0 = 10
        kappa = 0.75

        theta_prev = theta0
        samples = [theta0]

        stats = {'h_stat': [], 'epsilon': [], 'epsilon_bar': [], 'trajectory_length': []}

        for m in range(1, M+1):
            # Sample momentum
            r0 = np.random.normal(size=theta_prev.shape)

            # Sample slice variable
            joint_prob = self.L(theta_prev) - 0.5 * np.dot(r0, r0)
            log_u = joint_prob - np.random.exponential(1)

            # Initialize tree
            theta_minus = theta_prev
            theta_plus = theta_prev
            r_minus = r0
            r_plus = r0
            j = 0
            theta_next = theta_prev
            n = 1
            s = 1
            alpha = 0.0
            n_alpha = 0.0

            # Build the tree until the stopping criterion is met
            while s == 1 and j < self.max_depth:
                # Choose a direction
                v = np.random.choice([-1, 1])

                if v == -1:
                    theta_minus, r_minus, _, _, theta_prime, n_prime, s_prime, alpha, n_alpha = self.build_tree(theta_minus, r_minus, log_u, v, j, epsilon, theta_prev, r0, self.L, self.grad)

                else:
                    _, _, theta_plus, r_plus, theta_prime, n_prime, s_prime, alpha, n_alpha = self.build_tree(theta_plus, r_plus, log_u, v, j, epsilon, theta_prev, r0, self.L, self.grad)
                
                if s_prime == 1 and np.random.uniform() < min(1, n_prime / (n + 1e-10)):
                    theta_next = theta_prime
                    
                n += n_prime
                s = s_prime and int(np.dot(theta_plus - theta_minus, r_minus) >= 0) and int(np.dot(theta_plus - theta_minus, r_plus) >= 0)
                alpha += alpha
                n_alpha += n_alpha
                j += 1
            
            # Adapt epsilon using dual averaging
            if m <= M_adapt:
                H_bar = (1 - 1/(m + t0)) * H_bar + (1/(m + t0)) * (delta - alpha / n_alpha)
                log_epsilon = mu - (np.sqrt(m) / gamma) * H_bar
                epsilon = np.exp(log_epsilon)
                log_epsilon_bar = m ** (-kappa) * log_epsilon + (1 - m ** (-kappa)) * np.log(epsilon_bar)
                epsilon_bar = np.exp(log_epsilon_bar)
            
            else:
                epsilon = epsilon_bar

            theta_prev = theta_next
            samples.append(theta_prev)

            stats['h_stat'].append(alpha / n_alpha if n_alpha > 0 else 0.0)
            stats['epsilon'].append(epsilon)
            stats['epsilon_bar'].append(epsilon_bar)
            stats['trajectory_length'].append(2**(j-1))

        return np.array(samples), stats
