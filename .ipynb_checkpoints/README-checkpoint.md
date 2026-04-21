# ASTR4470FinalProject
## Configuration File (config.json)

- N: number of particles in the simulation
- G: gravitational constant (set to 1 for simplicity)
- epsilon: softening parameter to avoid singular forces
- theta: Barnes–Hut opening angle (controls accuracy)
- dt: time step size
- t_end: total simulation time
- output_interval: how often results are printed/saved
- initial_condition: type of initial particle setup
- mass_mode: "equal" or "random"
- mass_value: particle mass if using equal masses
- box_size: size of the simulation region
- random_seed: ensures reproducibility
- force_method: "direct" or "barnes_hut"