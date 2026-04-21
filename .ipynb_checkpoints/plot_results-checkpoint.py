import os
import numpy as np
import matplotlib.pyplot as plt


def make_figure_folder(folder_name="figures"):
    """
    Create the figure folder if it does not already exist.

    Parameters
    ----------
    folder_name : str
        Name of the folder where figures will be saved.
    """
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


def plot_energy_history(input_file="outputs/energy_history.txt",
                        output_file="figures/energy_history.png"):
    """
    Read the saved energy history file and make a plot of
    kinetic, potential, and total energy versus time.

    Parameters
    ----------
    input_file : str
        File containing time and energy data.
    output_file : str
        Name of the figure file to save.
    """

    data = np.loadtxt(input_file)

    times = data[:, 0]
    kinetic = data[:, 1]
    potential = data[:, 2]
    total = data[:, 3]

    plt.figure(figsize=(8, 6))
    plt.plot(times, kinetic, label="Kinetic Energy")
    plt.plot(times, potential, label="Potential Energy")
    plt.plot(times, total, label="Total Energy")

    plt.xlabel("Time")
    plt.ylabel("Energy")
    plt.title("Energy History of the Simulation")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(output_file, dpi=200)
    plt.close()


def plot_snapshot(snapshot_file="outputs/snapshot_00000.txt",
                  output_file="figures/snapshot_positions.png"):
    """
    Read one particle snapshot file and make a scatter plot
    of particle positions.

    Parameters
    ----------
    snapshot_file : str
        File containing one particle snapshot.
    output_file : str
        Name of the figure file to save.
    """

    data = np.loadtxt(snapshot_file)

    x = data[:, 1]
    y = data[:, 2]

    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, s=20)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Particle Positions")
    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(output_file, dpi=200)
    plt.close()


def plot_runtime_scaling(input_file="outputs/runtime_scaling.txt",
                         output_file="figures/runtime_scaling.png"):
    """
    Read the saved runtime scaling file and make a plot of
    runtime versus particle number for direct and Barnes-Hut methods.

    Parameters
    ----------
    input_file : str
        File containing particle count and runtime data.
    output_file : str
        Name of the figure file to save.
    """

    data = np.loadtxt(input_file)

    particle_counts = data[:, 0]
    direct_times = data[:, 1]
    barnes_hut_times = data[:, 2]

    plt.figure(figsize=(8, 6))
    plt.plot(particle_counts, direct_times, marker="o", label="Direct Summation")
    plt.plot(particle_counts, barnes_hut_times, marker="o", label="Barnes-Hut")

    plt.xlabel("Number of Particles")
    plt.ylabel("Runtime (s)")
    plt.title("Runtime Scaling Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(output_file, dpi=200)
    plt.close()


def main():
    """
    Make the standard plots for the simulation output.
    """

    make_figure_folder("figures")

    # Plot energy history
    plot_energy_history()

    # Plot the first saved particle snapshot
    plot_snapshot()

    # Plot runtime scaling results
    plot_runtime_scaling()

    print("Plots saved in the figures folder.")


if __name__ == "__main__":
    main()