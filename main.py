import torch
import timeit
import os
# import resource
import sys

import evolve_polarization

def main():

    print("\n==================== PHASE-FIELD SIMULATIONS =======================\n")

    # Start time
    start_tm = timeit.default_timer()

    # Device agnostic code
    if torch.cuda.is_available():
        device = torch.device("cuda")  # Use CUDA device
        print("CUDA is available. Using GPU.")
        print(torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")  # Use CPU
        print("CUDA is not available. Using CPU.")

    # ------------------------- SIMULATIONS PARAMETERS -------------------------
    grid_points = (120, 120, 1) # (Nx, Ny, Nz)
    grid_space = (5e-10, 5e-10, 5e-10) # (dx, dy, dz) in [m]

    # time parameters: nsteps - number of time steps, nt- time frame to save data
    time = (100, 10, 1e-13)   # (nsteps, nt, dt), dt is in [sec]

    # external electric field in each direction
    elec_field_ext = [0, 0, 0]  # (E_app_x, E_app_y, E_app_z) must be in V/m

    # external applied strain in each direction
    eps_ext_applied = [0, 0, 0] # (eps_app_x, eps_app_y, eps_app_z)

    # maximum electric field of the hysteresis loop
    E_max_loop = (0, 2) # (E, direction) E must be in V/m, direction: x=0, y=1, z=2

    # domain_type = "random" OR "90" OR "180" OR "minus_z"
    domain_type = "random"

    # saves the results, "YES" or "NO"
    save_data = "YES"

    # directory to save results
    FOLDER = os.getcwd()

    # Evolve
    evolve_polarization.Evolve_Sponteneous_Polarization_Isotropic(device, FOLDER, grid_points, grid_space, time,
                                    elec_field_ext, eps_ext_applied, domain_type, save_data)
    # ----------------------------------------------------------------------------
    stop_tm = timeit.default_timer()
    print("\n====================== SIMULATIONS  FINISHED =========================\n")
    print('Execution time: ' + str( format((stop_tm - start_tm)/60,'.3f')) + ' min' )
    print('\nSimulation parameters:')
    print('Grid points = ', grid_points)
    print('Grid space = ', grid_space)
    print('Time setup = ', time)

    # Get memory usage after execution in kilobytes
    # usage_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    # # Convert to megabytes and gigabytes for printing
    # usage_mb = usage_kb / 1024  # Convert to megabytes
    # usage_gb = usage_mb / 1024   # Convert to gigabytes

    # print(f"Memory usage: {usage_gb:.4f} GB")  # Print memory usage in GB with two decimal places
    print("========================================================================")

if __name__ == "__main__":
    main()
