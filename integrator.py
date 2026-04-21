from forces import compute_direct_accelerations
from barnes_hut import compute_barnes_hut_accelerations


def compute_accelerations(positions, masses, G, epsilon, method="direct", theta=None, box_size=None):
    """
    Compute particle accelerations using the chosen force method.

    Parameters
    ----------
    positions : ndarray, shape (N, 2)
        Current particle positions.
    masses : ndarray, shape (N,)
        Particle masses.
    G : float
        Gravitational constant.
    epsilon : float
        Softening parameter.
    method : str
        Force calculation method: "direct" or "barnes_hut".
    theta : float or None
        Barnes-Hut opening angle. Needed if method = "barnes_hut".
    box_size : float or None
        Half-size of the simulation box. Needed if method = "barnes_hut".

    Returns
    -------
    accelerations : ndarray, shape (N, 2)
        Acceleration of each particle.
    """

    if method == "direct":
        return compute_direct_accelerations(positions, masses, G, epsilon)

    elif method == "barnes_hut":
        if theta is None or box_size is None:
            raise ValueError("theta and box_size must be provided for Barnes-Hut method.")

        return compute_barnes_hut_accelerations(
            positions, masses, G, epsilon, theta, box_size
        )

    else:
        raise ValueError(f"Unknown force method: {method}")


def leapfrog_step(positions, velocities, masses, dt, G, epsilon,
                  method="direct", theta=None, box_size=None):
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
    method : str
        Force calculation method: "direct" or "barnes_hut".
    theta : float or None
        Barnes-Hut opening angle if needed.
    box_size : float or None
        Root box half-size if needed.

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
    accelerations_old = compute_accelerations(
        positions, masses, G, epsilon,
        method=method, theta=theta, box_size=box_size
    )

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
    accelerations_new = compute_accelerations(
        new_positions, masses, G, epsilon,
        method=method, theta=theta, box_size=box_size
    )

    # -----------------------------------------
    # 5. Kick: finish velocity update
    # -----------------------------------------
    new_velocities = v_half + 0.5 * accelerations_new * dt

    return new_positions, new_velocities