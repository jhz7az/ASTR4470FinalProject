# ASTR4470 Final Project

## Project Title
A 2D Barnes-Hut Tree Code for Gravitational N-Body Simulations

---

## Project Description
This project implements a two-dimensional gravitational N-body simulation in Python.

The code computes gravitational accelerations using:
- direct summation (exact)
- Barnes-Hut tree algorithm (approximate)

The system is evolved forward in time using a leapfrog integrator.

Additional features:
- Computes kinetic, potential, and total energy
- Saves simulation outputs to files
- Includes scripts for plotting and testing

Scientific goal:
Compare the accuracy and computational performance of the Barnes-Hut approximation against direct summation.

---

## Files Included

### Main Simulation Files
- main.py  
  Runs the N-body simulation

- config.json  
  Input parameters for the simulation

- forces.py  
  Direct-summation gravitational forces

- barnes_hut.py  
  Barnes-Hut force calculation

- quadtree.py  
  Quadtree data structure

- integrator.py  
  Leapfrog time integration

- diagnostics.py  
  Energy calculations

- io_utils.py  
  Output handling (snapshots, energy logs)

---

### Analysis and Testing Files
- plot_results.py  
  Generates plots from simulation outputs

- run_tests.py  
  Runs:
  - Force symmetry test  
  - Energy conservation test  
  - Barnes-Hut vs direct comparison  

- runtime_test.py  
  Measures runtime scaling

---

## Required Python Packages
- numpy
- matplotlib
- json (standard library)
- time (standard library)
- os (standard library)

---

## How to Run the Main Simulation
1. Edit parameters in config.json  
2. Run:
   python main.py  

This will:
- Initialize particles
- Run the simulation
- Save outputs in the outputs/ folder:
  - snapshots
  - energy history

---

## Important Input Parameters (config.json)

N : number of particles  
G : gravitational constant  
epsilon : softening parameter  
theta : Barnes-Hut opening angle  
dt : timestep size  
t_end : total simulation time  
output_interval : output frequency  
mass_mode : "equal" or variable  
mass_value : particle mass if equal  
box_size : simulation domain size  
random_seed : reproducibility seed  
force_method : "direct" or "barnes_hut"  

---

## How to Run the Tests
Run:
python run_tests.py  

This runs:
- force symmetry test  
- energy conservation test  
- Barnes-Hut vs direct comparison  

---

## How to Run Runtime Comparison
Run:
python runtime_test.py  

Output:
outputs/runtime_scaling.txt  

---

## How to Make Plots
Run:
python plot_results.py  

This creates figures in the figures/ folder:
- energy_history.png  
- snapshot_positions.png  
- runtime_scaling.png  

---

## Output Files

In outputs/:
- snapshot_XXXXX.txt  
- energy_history.txt  
- runtime_scaling.txt  

In figures/:
- energy_history.png  
- snapshot_positions.png  
- runtime_scaling.png  

---

## Notes
- Direct method is exact and used as a baseline  
- Barnes-Hut is approximate but faster for large N  
- For small N, Barnes-Hut may be slower due to tree overhead  
- Reducing dt improves energy conservation  

---

## Author
Malhar Kulkarni  

## Course
ASTR 4470: Computational Astronomy/Astrophysics