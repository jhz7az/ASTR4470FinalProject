import json
import numpy as np
from integrator import leapfrog_step
from diagnostics import kinetic_energy, potential_energy, total_energy
from io_utils import make_output_folder, save_snapshot, save_energy_history


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
    output_interval = config["output_interval"]  # how often to print/save output
    theta = config["theta"]              # Barnes-Hut opening angle
    force_method = config["force_method"]  # "direct" or "barnes_hut"
    
    
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
    # 6. Set up output folder and diagnostic storage
    # --------------------------------------------------
    # Create an output folder if it does not already exist
    make_output_folder("outputs")

    # These lists will store the energy history over the run
    times = []
    kinetic_list = []
    potential_list = []
    total_list = []

    # --------------------------------------------------
    # 7. Set up the time loop
    # --------------------------------------------------
    # t keeps track of the current simulation time
    # step keeps track of the number of timesteps completed
    t = 0.0
    step = 0

    # --------------------------------------------------
    # 8. Main simulation loop
    # --------------------------------------------------
    while t < t_end:
        # Advance the system by one timestep using leapfrog
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
        
        # Compute diagnostic quantities
        K = kinetic_energy(velocities, masses)
        U = potential_energy(positions, masses, G, epsilon)
        E = total_energy(positions, velocities, masses, G, epsilon)

        # Store diagnostic quantities for later output
        times.append(t)
        kinetic_list.append(K)
        potential_list.append(U)
        total_list.append(E)

        # Print progress and save a snapshot every few steps
        if step % output_interval == 0:
            print(f"Step {step}, time = {t:.3f}, K = {K:.4f}, U = {U:.4f}, E = {E:.4f}")

            snapshot_name = f"outputs/snapshot_{step:05d}.txt"
            save_snapshot(snapshot_name, positions, velocities, masses, t)

        # Move to the next timestep
        t += dt
        step += 1

    # --------------------------------------------------
    # 9. Save the full energy history after the run ends
    # --------------------------------------------------
    save_energy_history(
        "outputs/energy_history.txt",
        times,
        kinetic_list,
        potential_list,
        total_list
    )

    # --------------------------------------------------
    # 10. End of simulation
    # --------------------------------------------------
    print()
    print("Simulation finished.")


# This makes sure main() runs only when this file is executed directly
if __name__ == "__main__":
    main()