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

    # Load the columns from the text file
    data = np.loadtxt(input_file)

    # Columns are: time, kinetic, potential, total
    times = data[:, 0]
    kinetic = data[:, 1]
    potential = data[:, 2]
    total = data[:, 3]

    # Make the plot
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

    # Save the figure
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

    # Load the snapshot data
    # Columns are: mass, x, y, vx, vy
    data = np.loadtxt(snapshot_file)

    x = data[:, 1]
    y = data[:, 2]

    # Make the scatter plot
    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, s=20)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Particle Positions")
    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()

    # Save the figure
    plt.savefig(output_file, dpi=200)
    plt.close()


def main():
    """
    Make the standard plots for the simulation output.
    """

    make_figure_folder("figures")

    # Plot energy history from the full run
    plot_energy_history()

    # Plot the first saved snapshot
    plot_snapshot()

    print("Plots saved in the figures folder.")


if __name__ == "__main__":
    main()