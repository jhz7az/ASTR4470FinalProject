import numpy as np


def compute_direct_accelerations(positions, masses, G, epsilon):
    """
    Compute the gravitational acceleration on every particle
    using direct summation.

    Parameters
    ----------
    positions : ndarray, shape (N, 2)
        Array of particle positions. Each row is [x, y].
    masses : ndarray, shape (N,)
        Array of particle masses.
    G : float
        Gravitational constant.
    epsilon : float
        Softening parameter to prevent the force from blowing up
        at very small separations.

    Returns
    -------
    accelerations : ndarray, shape (N, 2)
        Gravitational acceleration on each particle.
    """

    # Number of particles
    N = len(masses)

    # Create an array to store the acceleration of each particle
    accelerations = np.zeros((N, 2))

    # Loop over each particle i
    for i in range(N):
        # Start the total acceleration on particle i at zero
        ax = 0.0
        ay = 0.0

        # Sum contributions from all other particles j
        for j in range(N):
            # A particle should not exert a force on itself
            if i == j:
                continue

            # Difference in position between particles j and i
            dx = positions[j, 0] - positions[i, 0]
            dy = positions[j, 1] - positions[i, 1]

            # Squared distance with softening included
            r2 = dx**2 + dy**2 + epsilon**2

            # Compute 1 / r^3 term
            r3 = r2**1.5

            # Add the contribution from particle j
            ax += G * masses[j] * dx / r3
            ay += G * masses[j] * dy / r3

        # Store the total acceleration for particle i
        accelerations[i, 0] = ax
        accelerations[i, 1] = ay

    return accelerations