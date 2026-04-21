import os
import numpy as np


def make_output_folder(folder_name="outputs"):
    """
    Create the output folder if it does not already exist.

    Parameters
    ----------
    folder_name : str
        Name of the folder where output files will be stored.
    """

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


def save_snapshot(filename, positions, velocities, masses, time):
    """
    Save one simulation snapshot to a text file.

    Parameters
    ----------
    filename : str
        Name of the file to write.
    positions : ndarray, shape (N, 2)
        Particle positions.
    velocities : ndarray, shape (N, 2)
        Particle velocities.
    masses : ndarray, shape (N,)
        Particle masses.
    time : float
        Current simulation time.
    """

    # Number of particles
    N = len(masses)

    # Create an array with one row per particle
    # Columns: mass, x, y, vx, vy
    data = np.zeros((N, 5))
    data[:, 0] = masses
    data[:, 1:3] = positions
    data[:, 3:5] = velocities

    # Header line for clarity
    header = f"time = {time:.6f}\ncolumns: mass x y vx vy"

    # Save as plain text
    np.savetxt(filename, data, header=header)


def save_energy_history(filename, times, kinetic, potential, total):
    """
    Save the energy history of the simulation to a text file.

    Parameters
    ----------
    filename : str
        Name of the file to write.
    times : list or ndarray
        Times at which energies were recorded.
    kinetic : list or ndarray
        Kinetic energy values.
    potential : list or ndarray
        Potential energy values.
    total : list or ndarray
        Total energy values.
    """

    # Stack all columns together into one array
    data = np.column_stack((times, kinetic, potential, total))

    # Header line describing the columns
    header = "time kinetic_energy potential_energy total_energy"

    # Save as plain text
    np.savetxt(filename, data, header=header)