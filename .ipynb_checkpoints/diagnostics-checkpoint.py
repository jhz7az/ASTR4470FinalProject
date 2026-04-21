import numpy as np


def kinetic_energy(velocities, masses):
    """
    Compute the total kinetic energy of the system.

    Parameters
    ----------
    velocities : ndarray, shape (N, 2)
        Array of particle velocities.
    masses : ndarray, shape (N,)
        Array of particle masses.

    Returns
    -------
    K : float
        Total kinetic energy.
    """

    # Speed squared for each particle: vx^2 + vy^2
    speed_squared = velocities[:, 0]**2 + velocities[:, 1]**2

    # K = 1/2 sum(m v^2)
    K = 0.5 * np.sum(masses * speed_squared)

    return K


def potential_energy(positions, masses, G, epsilon):
    """
    Compute the total gravitational potential energy of the system
    using direct pairwise summation.

    Parameters
    ----------
    positions : ndarray, shape (N, 2)
        Array of particle positions.
    masses : ndarray, shape (N,)
        Array of particle masses.
    G : float
        Gravitational constant.
    epsilon : float
        Softening parameter.

    Returns
    -------
    U : float
        Total gravitational potential energy.
    """

    N = len(masses)
    U = 0.0

    # Loop over all unique particle pairs
    # We use j > i so each pair is counted only once
    for i in range(N):
        for j in range(i + 1, N):
            # Separation between particles i and j
            dx = positions[j, 0] - positions[i, 0]
            dy = positions[j, 1] - positions[i, 1]

            # Softened distance
            r = np.sqrt(dx**2 + dy**2 + epsilon**2)

            # Add pair contribution to potential energy
            U += -G * masses[i] * masses[j] / r

    return U


def total_energy(positions, velocities, masses, G, epsilon):
    """
    Compute the total energy of the system.

    Parameters
    ----------
    positions : ndarray, shape (N, 2)
        Array of particle positions.
    velocities : ndarray, shape (N, 2)
        Array of particle velocities.
    masses : ndarray, shape (N,)
        Array of particle masses.
    G : float
        Gravitational constant.
    epsilon : float
        Softening parameter.

    Returns
    -------
    E : float
        Total energy = kinetic + potential.
    """

    K = kinetic_energy(velocities, masses)
    U = potential_energy(positions, masses, G, epsilon)
    E = K + U

    return E