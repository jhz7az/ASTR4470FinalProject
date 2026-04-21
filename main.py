import json
import numpy as np
from forces import compute_direct_accelerations
from integrator import leapfrog_step


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


def main():
    # --------------------------------------------------
    # 1. Load all simulation parameters from config.json
    # --------------------------------------------------
    config = load_config("config.json")

    # Pull out the main parameters so they are easy to use later
    N = config["N"]                  # number of particles
    G = config["G"]                  # gravitational constant
    epsilon = config["epsilon"]      # softening parameter
    dt = config["dt"]                # timestep size
    t_end = config["t_end"]          # total simulation time
    output_interval = config["output_interval"]  # how often to print progress

    # Print a short summary so we can confirm the code read the input correctly
    print("Simulation parameters:")
    print(f"N = {N}")
    print(f"G = {G}")
    print(f"epsilon = {epsilon}")
    print(f"dt = {dt}")
    print(f"t_end = {t_end}")
    print()

    # --------------------------------------------------
    # 2. Set the random seed so results are reproducible
    # --------------------------------------------------
    np.random.seed(config["random_seed"])

    # --------------------------------------------------
    # 3. Initialize particle positions
    # --------------------------------------------------
    # For now, we place particles randomly in a square box:
    # x and y each run from -box_size to +box_size
    box_size = config["box_size"]
    positions = np.random.uniform(-box_size, box_size, size=(N, 2))

    # --------------------------------------------------
    # 4. Initialize particle velocities
    # --------------------------------------------------
    # For this first version, start all velocities at zero
    velocities = np.zeros((N, 2))

    # --------------------------------------------------
    # 5. Initialize particle masses
    # --------------------------------------------------
    # If mass_mode is "equal", all particles get the same mass.
    # Otherwise, give them random masses in a simple range.
    if config["mass_mode"] == "equal":
        masses = np.ones(N) * config["mass_value"]
    else:
        masses = np.random.uniform(0.5, 1.5, size=N)

    print("Particles initialized.")
    print(f"positions shape = {positions.shape}")
    print(f"velocities shape = {velocities.shape}")
    print(f"masses shape = {masses.shape}")
    print()

    # --------------------------------------------------
    # 6. Set up the time loop
    # --------------------------------------------------
    # t keeps track of the current simulation time
    # step keeps track of the number of timesteps completed
    t = 0.0
    step = 0

    # --------------------------------------------------
    # 7. Main simulation loop
    # --------------------------------------------------
    # Right now this is just a skeleton loop.
    # We are not computing real gravity yet.
    # For now, accelerations are all set to zero.
    while t < t_end:
        # Placeholder acceleration array
        # This has the same shape as positions: one ax, ay pair per particle
        positions, velocities = leapfrog_step(positions, velocities, masses, dt, G, epsilon)

        # Print progress every few steps
        if step % output_interval == 0:
            print(f"Step {step}, time = {t:.3f}")

        # Move to the next timestep
        t += dt
        step += 1

    # --------------------------------------------------
    # 8. End of simulation
    # --------------------------------------------------
    print()
    print("Simulation finished.")


# This makes sure main() runs only when this file is executed directly
if __name__ == "__main__":
    main()