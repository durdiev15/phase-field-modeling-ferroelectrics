import torch
import timeit
import os
# import resource
import sys

from evolve_polarization import Evolve_Sponteneous_Polarization_Isotropic

def main():

    print("\n==================== PHASE-FIELD SIMULATIONS =======================\n")

    # Device agnostic code
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------- SIMULATIONS PARAMETERS -------------------------
    # Add the simulation parameters into sim_params dictionary 
    c_tet = 4.032                         # angstrom
    a_tet = 3.992                         # angstrom
    sim_dict = {

        # Phase-field parameters:
        'G': 12e-3,                       # J/m²=N/m Interfacial Energy
        'l': 1.5E-9,                      # m: Thickness of interface: Length Scale
        'P0': 0.26,                        # C/m²: Maximum polarization
        'mob': 26/75*1e3,                   # A/Vm: mobility beta^-1
        'c_tet': c_tet,
        'a_tet': a_tet,
        'eps_spon0':   2*(c_tet-a_tet)/(c_tet+2*a_tet), # spontaneous strain

        # Landau polynomial coeffs (dimensionless):
        'a1': -86/75,
        'a2': -53/75,
        'a3': 134/25,
        'a4': 64/75,

        # Grad and Sep coeffs to adjust the DW energy and width:
        'k_sep':  0.70,   
        'k_grad': 0.35,  

        # Piezoelectric tensor in C/m²:
        'e31': -0.7,  # C/m²
        'e33': 6.7,
        'e15': 34.2,

        # Elastic tensor in N/m²
        'C11': 22.2e10,
        'C12': 11.1e10,
        'C44': 6.1e10,

        # dielectric constant in C/(Vm)
        'k': 19.5e-9,

        # Grid parameters:
        'Nx': 120, 'Ny': 120, 'Nz': 1,
        'dx': 5e-10, 'dy': 5e-10, 'dz': 5e-10, # spacing in m
        
        # Simulation time parameters:
        'nsteps': 5000,  # total number of time steps
        'nt': 100,       # time interval to save data
        'dt': 1e-13,      # time step in s

        # Aplpied electric field in V/m:
        'E_ext_1': 0, 'E_ext_2': 0, 'E_ext_3': 0,

        # Max electric field for P-E loop:
        'E_max': 0, # in V/m
        'E_direc': 2, # in whic direction 0-<100>, 1-<010>, 2-<001> 

        # Applied strain:
        'eps_ext_11': 0, 'eps_ext_22': 0, 'eps_ext_33': 0,
        'eps_ext_12': 0, 'eps_ext_13': 0, 'eps_ext_23': 0,

        # Domain type:
        'domain': "random",

    }

    # directory to save results
    FOLDER = os.getcwd() + "/results"
    HDF_RESULTS_FILE = FOLDER + "/results.h5"

    # check if results.h5 exists 
    if os.path.exists(HDF_RESULTS_FILE):
        os.remove(HDF_RESULTS_FILE)  # Remove the file

    # Evolve
    Evolve_Sponteneous_Polarization_Isotropic(device, FOLDER, HDF_RESULTS_FILE, sim_dict)

if __name__ == "__main__":
    main()
