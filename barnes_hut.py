import numpy as np
from quadtree import build_quadtree


def compute_acceleration_from_node(node, particle_index, positions, masses, G, epsilon, theta):
    """
    Compute the gravitational acceleration on one particle due to one quadtree node.

    This function is recursive. It decides whether to:
    1. approximate the whole node as one pseudo-particle, or
    2. open the node and examine its children.

    Parameters
    ----------
    node : QuadTreeNode
        Current quadtree node.
    particle_index : int
        Index of the particle whose acceleration we are computing.
    positions : ndarray, shape (N, 2)
        Array of particle positions.
    masses : ndarray, shape (N,)
        Array of particle masses.
    G : float
        Gravitational constant.
    epsilon : float
        Softening parameter.
    theta : float
        Barnes-Hut opening angle.

    Returns
    -------
    ax, ay : float
        Acceleration components on the chosen particle.
    """

    # If this node has no mass, it contributes nothing
    if node.mass == 0.0:
        return 0.0, 0.0

    # Position of the particle we care about
    x = positions[particle_index, 0]
    y = positions[particle_index, 1]

    # ------------------------------------------
    # Case 1: leaf node
    # ------------------------------------------
    if node.is_leaf():
        # If the leaf is empty, no contribution
        if node.particle_index is None:
            return 0.0, 0.0

        # A particle should not act on itself
        if node.particle_index == particle_index:
            return 0.0, 0.0

        # Otherwise compute the direct contribution from the single particle
        dx = node.com_x - x
        dy = node.com_y - y

        r2 = dx**2 + dy**2 + epsilon**2
        r3 = r2**1.5

        ax = G * node.mass * dx / r3
        ay = G * node.mass * dy / r3

        return ax, ay

    # ------------------------------------------
    # Case 2: internal node
    # Decide whether to approximate this node
    # or open it further
    # ------------------------------------------
    dx = node.com_x - x
    dy = node.com_y - y
    d = np.sqrt(dx**2 + dy**2 + epsilon**2)

    # Full width of this square cell
    s = 2.0 * node.half_size

    # Barnes-Hut opening-angle criterion
    # If s / d < theta, the cell is far enough away
    # that we approximate it as one pseudo-particle
    if s / d < theta:
        r2 = dx**2 + dy**2 + epsilon**2
        r3 = r2**1.5

        ax = G * node.mass * dx / r3
        ay = G * node.mass * dy / r3

        return ax, ay

    # Otherwise, open the node and sum contributions from children
    ax_total = 0.0
    ay_total = 0.0

    for child in node.children:
        ax_child, ay_child = compute_acceleration_from_node(
            child, particle_index, positions, masses, G, epsilon, theta
        )
        ax_total += ax_child
        ay_total += ay_child

    return ax_total, ay_total


def compute_barnes_hut_accelerations(positions, masses, G, epsilon, theta, box_size):
    """
    Compute gravitational accelerations for all particles using the Barnes-Hut method.

    Parameters
    ----------
    positions : ndarray, shape (N, 2)
        Particle positions.
    masses : ndarray, shape (N,)
        Particle masses.
    G : float
        Gravitational constant.
    epsilon : float
        Softening parameter.
    theta : float
        Barnes-Hut opening angle.
    box_size : float
        Half-size of the root simulation box.

    Returns
    -------
    accelerations : ndarray, shape (N, 2)
        Barnes-Hut accelerations for all particles.
    """

    # Build the quadtree for the current particle distribution
    root = build_quadtree(positions, masses, box_size)

    N = len(masses)
    accelerations = np.zeros((N, 2))

    # Compute acceleration on each particle by traversing the tree
    for i in range(N):
        ax, ay = compute_acceleration_from_node(
            root, i, positions, masses, G, epsilon, theta
        )
        accelerations[i, 0] = ax
        accelerations[i, 1] = ay

    return accelerations