import numpy as np
from forces import compute_direct_accelerations


def leapfrog_step(positions, velocities, masses, dt, G, epsilon):
    """
    Advance the system by one timestep using the leapfrog method.

    Parameters
    ----------
    positions : ndarray, shape (N, 2)
        Current particle positions.
    velocities : ndarray, shape (N, 2)
        Current particle velocities.
    masses : ndarray, shape (N,)
        Particle masses.
    dt : float
        Timestep size.
    G : float
        Gravitational constant.
    epsilon : float
        Softening parameter.

    Returns
    -------
    new_positions : ndarray, shape (N, 2)
        Updated particle positions after one timestep.
    new_velocities : ndarray, shape (N, 2)
        Updated particle velocities after one timestep.
    """

    # -----------------------------------------
    # 1. Compute acceleration at current time
    # -----------------------------------------
    accelerations_old = compute_direct_accelerations(positions, masses, G, epsilon)

    # -----------------------------------------
    # 2. Kick: update velocity by half step
    # -----------------------------------------
    v_half = velocities + 0.5 * accelerations_old * dt

    # -----------------------------------------
    # 3. Drift: update position by full step
    # -----------------------------------------
    new_positions = positions + v_half * dt

    # -----------------------------------------
    # 4. Compute acceleration at new position
    # -----------------------------------------
    accelerations_new = compute_direct_accelerations(new_positions, masses, G, epsilon)

    # -----------------------------------------
    # 5. Kick: finish velocity update
    # -----------------------------------------
    new_velocities = v_half + 0.5 * accelerations_new * dt

    return new_positions, new_velocities