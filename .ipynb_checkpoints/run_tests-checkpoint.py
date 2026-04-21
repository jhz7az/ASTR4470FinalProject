import json
import numpy as np

from forces import compute_direct_accelerations
from barnes_hut import compute_barnes_hut_accelerations
from integrator import leapfrog_step
from diagnostics import total_energy


def load_config(filename):
    """
    Read the simulation settings from a JSON file.

    Parameters
    ----------
    filename : str
        Name of the JSON configuration file.

    Returns
    -------
    config : dict
        Dictionary containing all simulation parameters.
    """
    with open(filename, "r") as f:
        config = json.load(f)
    return config


def test_force_symmetry():
    """
    Test the direct force calculation on a simple two-particle system.

    We place two equal-mass particles symmetrically on the x-axis.
    Their accelerations should have equal magnitude and opposite direction.
    """

    print("Running force symmetry test...")

    positions = np.array([
        [-1.0, 0.0],
        [1.0, 0.0]
    ])

    masses = np.array([1.0, 1.0])
    G = 1.0
    epsilon = 0.01

    accelerations = compute_direct_accelerations(positions, masses, G, epsilon)

    a1 = accelerations[0]
    a2 = accelerations[1]

    print("Particle 1 acceleration:", a1)
    print("Particle 2 acceleration:", a2)

    if np.allclose(a1, -a2, atol=1e-10):
        print("Force symmetry test PASSED.\n")
        return True
    else:
        print("Force symmetry test FAILED.\n")
        return False


def test_energy_conservation():
    """
    Run a short simulation and check whether total energy
    stays approximately constant.
    """

    print("Running energy conservation test...")

    config = load_config("config.json")

    N = config["N"]
    G = config["G"]
    epsilon = config["epsilon"]
    dt = config["dt"]
    t_end = config["t_end"]
    theta = config["theta"]
    box_size = config["box_size"]
    force_method = config["force_method"]

    np.random.seed(config["random_seed"])

    positions = np.random.uniform(-box_size, box_size, size=(N, 2))
    velocities = np.zeros((N, 2))

    if config["mass_mode"] == "equal":
        masses = np.ones(N) * config["mass_value"]
    else:
        masses = np.random.uniform(0.5, 1.5, size=N)

    E_initial = total_energy(positions, velocities, masses, G, epsilon)

    t = 0.0
    while t < t_end:
        positions, velocities = leapfrog_step(
            positions,
            velocities,
            masses,
            dt,
            G,
            epsilon,
            method=force_method,
            theta=theta,
            box_size=box_size
        )
        t += dt

    E_final = total_energy(positions, velocities, masses, G, epsilon)

    fractional_change = abs((E_final - E_initial) / E_initial)

    print(f"Initial total energy = {E_initial:.6f}")
    print(f"Final total energy   = {E_final:.6f}")
    print(f"Fractional change    = {fractional_change:.6e}")

    if fractional_change < 0.1:
        print("Energy conservation test PASSED.\n")
        return True
    else:
        print("Energy conservation test FAILED.\n")
        return False


def test_barnes_hut_vs_direct():
    """
    Compare Barnes-Hut accelerations against direct-summation accelerations
    for a modest number of particles.

    This checks whether Barnes-Hut gives results reasonably close to
    the exact direct method.
    """

    print("Running Barnes-Hut vs direct comparison test...")

    # Use a modest particle number for comparison
    N = 30
    G = 1.0
    epsilon = 0.05
    theta = 0.5
    box_size = 5.0

    np.random.seed(42)

    positions = np.random.uniform(-box_size, box_size, size=(N, 2))
    masses = np.ones(N)

    # Compute exact direct accelerations
    direct_acc = compute_direct_accelerations(positions, masses, G, epsilon)

    # Compute Barnes-Hut accelerations
    bh_acc = compute_barnes_hut_accelerations(positions, masses, G, epsilon, theta, box_size)

    # Compute relative error
    diff = np.linalg.norm(direct_acc - bh_acc)
    direct_norm = np.linalg.norm(direct_acc)

    if direct_norm == 0.0:
        relative_error = diff
    else:
        relative_error = diff / direct_norm

    print(f"Relative acceleration error = {relative_error:.6e}")

    # Loose but reasonable tolerance for a first Barnes-Hut test
    if relative_error < 0.2:
        print("Barnes-Hut comparison test PASSED.\n")
        return True
    else:
        print("Barnes-Hut comparison test FAILED.\n")
        return False


def main():
    """
    Run all implemented tests and report the results.
    """

    print("===================================")
    print("Running ASTR4470 Final Project Tests")
    print("===================================\n")

    results = []

    results.append(test_force_symmetry())
    results.append(test_energy_conservation())
    results.append(test_barnes_hut_vs_direct())

    n_passed = sum(results)
    n_total = len(results)

    print("===================================")
    print(f"Passed {n_passed} out of {n_total} tests.")
    print("===================================")


if __name__ == "__main__":
    main()