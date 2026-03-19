import numpy as np


def leapfrog(theta, r, epsilon, grad):
    """Performs a leapfrog step in Hamiltonian Monte Carlo."""
    # Update momentum by half step
    r_tilde = r + (epsilon / 2) * grad(theta)

    # Update position by full step
    theta_tilde = theta + epsilon * r_tilde

    # Update momentum by half step
    r_tilde = r_tilde + (epsilon / 2) * grad(theta_tilde)

    return theta_tilde, r_tilde


def find_reasonable_epsilon(theta, grad, L):
    """Heuristic for choosing an initial value of epsilon."""
    epsilon = 1.0
    r = np.random.normal(size=theta.shape)
    theta_prime, r_prime = leapfrog(theta, r, epsilon, grad)
    
    log_p = L(theta) - 0.5 * np.dot(r, r)
    log_p_prime = L(theta_prime) - 0.5 * np.dot(r_prime, r_prime)

    # Acceptance probability in log space
    log_a = log_p_prime - log_p
    a = 1 if log_a > np.log(0.5) else -1

    while (a * log_a) > -a * np.log(2):
        epsilon *= 2.0 ** a
        theta_prime, r_prime = leapfrog(theta, r, epsilon, grad)
        
        log_p_prime = L(theta_prime) - 0.5 * np.dot(r_prime, r_prime)
        log_a = log_p_prime - log_p

    return epsilon
