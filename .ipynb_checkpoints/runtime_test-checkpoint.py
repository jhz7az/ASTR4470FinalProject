import time
import numpy as np

from forces import compute_direct_accelerations
from barnes_hut import compute_barnes_hut_accelerations
from io_utils import make_output_folder


def run_runtime_test():
    """
    Compare the runtime of direct summation and Barnes-Hut
    force calculations for different particle numbers.
    """

    print("Running runtime scaling test...")

    # --------------------------------------------------
    # 1. Set fixed simulation parameters
    # --------------------------------------------------
    G = 1.0
    epsilon = 0.05
    theta = 0.5
    box_size = 5.0

    # Particle numbers to test
    particle_counts = [20, 50, 100, 200, 400]

    # Lists to store results
    direct_times = []
    barnes_hut_times = []

    # Make sure the output folder exists
    make_output_folder("outputs")

    # --------------------------------------------------
    # 2. Loop over different particle counts
    # --------------------------------------------------
    for N in particle_counts:
        print(f"Testing N = {N}")

        # Use a fixed seed so results are reproducible
        np.random.seed(42)

        # Generate positions and masses
        positions = np.random.uniform(-box_size, box_size, size=(N, 2))
        masses = np.ones(N)

        # ------------------------------------------
        # Direct summation timing
        # ------------------------------------------
        start_direct = time.time()
        compute_direct_accelerations(positions, masses, G, epsilon)
        end_direct = time.time()

        direct_runtime = end_direct - start_direct
        direct_times.append(direct_runtime)

        # ------------------------------------------
        # Barnes-Hut timing
        # ------------------------------------------
        start_bh = time.time()
        compute_barnes_hut_accelerations(positions, masses, G, epsilon, theta, box_size)
        end_bh = time.time()

        bh_runtime = end_bh - start_bh
        barnes_hut_times.append(bh_runtime)

        print(f"  Direct time     = {direct_runtime:.6f} s")
        print(f"  Barnes-Hut time = {bh_runtime:.6f} s")
        print()

    # --------------------------------------------------
    # 3. Save results to a text file
    # --------------------------------------------------
    output_data = np.column_stack((particle_counts, direct_times, barnes_hut_times))

    header = "N direct_runtime barnes_hut_runtime"
    np.savetxt("outputs/runtime_scaling.txt", output_data, header=header)

    print("Runtime test complete.")
    print("Results saved to outputs/runtime_scaling.txt")


if __name__ == "__main__":
    run_runtime_test()